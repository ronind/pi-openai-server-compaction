import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import test from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { WebSocketServer } from "ws";
import { normalizeContext } from "@earendil-works/pi-ai";
import { getModel } from "@earendil-works/pi-ai/compat";
import { createOpenAIWebSocketStreamFn, releaseWsSession } from "../src/openai-ws-stream.ts";

async function withServer(run) {
  const server = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await once(server, "listening");
  server.on("headers", (headers) => headers.push("x-test-observation: present"));
  const sessionId = randomUUID();
  const model = getModel("openai", "gpt-5.4-nano");
  const context = normalizeContext({ messages: [{ role: "user", content: "Hello", timestamp: 0 }] });
  const stream = createOpenAIWebSocketStreamFn({ url: `ws://127.0.0.1:${server.address().port}` });
  const options = { sessionId, apiKey: "offline-test", transport: "websocket", openaiWsWarmup: false };
  try {
    await run({ server, model, context, stream, options });
  } finally {
    releaseWsSession(sessionId);
    for (const socket of server.clients) socket.terminate();
    await new Promise((resolve) => server.close(resolve));
  }
}

function completed(id = "response-test") {
  return { type: "response.completed", response: {
    id, object: "response", created_at: 0, status: "completed", model: "gpt-5.4-nano",
    output: [{ type: "message", id: "message-test", role: "assistant", content: [{ type: "output_text", text: "Hello" }] }],
    usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
  } };
}

test("WebSocket stream observes response headers and awaits provider events in wire order", { timeout: 5000 }, async () => {
  await withServer(async ({ server, model, context, stream, options }) => {
    const observations = [];
    let request;
    server.on("connection", (socket) => socket.on("message", (data) => {
      request = JSON.parse(data.toString());
      socket.send(JSON.stringify({ type: "response.created", response: { id: "response-test", output: [] } }));
      socket.send(JSON.stringify({ type: "response.output_text.delta", delta: "Hello" }));
      socket.send(JSON.stringify(completed()));
    }));
    const events = [];
    const result = stream(model, context, {
      ...options,
      async onResponse(response, observedModel) {
        assert.equal(observedModel, model);
        assert.equal(response.status, 101);
        assert.equal(response.headers["x-test-observation"], "present");
        observations.push("response");
        await delay(5);
        observations.push("response:done");
      },
      onPayload(payload) {
        assert.equal(observations.at(-1), "response:done");
        return { ...payload, metadata: { observed: "yes" } };
      },
      async onProviderStreamEvent(event, observedModel) {
        assert.equal(observedModel, model);
        observations.push(event.type);
        await delay(event.type === "response.created" ? 10 : 1);
        observations.push(`${event.type}:done`);
      },
    });
    for await (const event of result) {
      events.push(event);
      if (event.type === "text_delta") assert.ok(observations.includes("response.output_text.delta:done"));
      if (event.type === "done") assert.ok(observations.includes("response.completed:done"));
    }
    assert.equal((await result.result()).stopReason, "stop");
    assert.equal(request.metadata.observed, "yes");
    assert.deepEqual(observations, ["response", "response:done", "response.created", "response.created:done",
      "response.output_text.delta", "response.output_text.delta:done", "response.completed", "response.completed:done"]);
    assert.equal(events.filter((event) => event.type === "done").length, 1);
  });
});

test("a provider observer failure ends the WebSocket stream with one error", { timeout: 5000 }, async () => {
  await withServer(async ({ server, model, context, stream, options }) => {
    server.on("connection", (socket) => socket.on("message", () => {
      socket.send(JSON.stringify({ type: "response.output_text.delta", delta: "Hello" }));
      socket.send(JSON.stringify(completed()));
    }));
    const result = stream(model, context, { ...options, onProviderStreamEvent() { throw new Error("observer failed"); } });
    const events = [];
    for await (const event of result) events.push(event);
    assert.equal((await result.result()).stopReason, "error");
    assert.match((await result.result()).errorMessage, /observer failed/);
    assert.equal(events.filter((event) => event.type === "error").length, 1);
    assert.equal(events.filter((event) => event.type === "done").length, 0);
  });
});

test("a received completion settles before a subsequent socket close", { timeout: 5000 }, async () => {
  await withServer(async ({ server, model, context, stream, options }) => {
    server.on("connection", (socket) => socket.on("message", () => {
      socket.send(JSON.stringify(completed()));
      socket.close(1000, "Response complete");
    }));
    const result = stream(model, context, { ...options, async onProviderStreamEvent() { await delay(50); } });
    const events = [];
    for await (const event of result) events.push(event);
    const message = await result.result();
    assert.equal(message.stopReason, "stop", message.errorMessage);
    assert.equal(message.usage.totalTokens, 2);
    assert.equal(events.filter((event) => event.type === "done").length, 1);
    assert.equal(events.filter((event) => event.type === "error").length, 0);
  });
});

test("a nonterminal socket close drains earlier provider observations before failing", { timeout: 5000 }, async () => {
  await withServer(async ({ server, model, context, stream, options }) => {
    server.on("connection", (socket) => socket.on("message", () => {
      socket.send(JSON.stringify({ type: "response.output_text.delta", delta: "Partial" }));
      socket.close(1000, "No completion");
    }));
    const observed = [];
    const result = stream(model, context, { ...options, async onProviderStreamEvent(event) {
      await delay(50);
      observed.push(event.type);
    } });
    const events = [];
    for await (const event of result) {
      if (event.type === "error") assert.deepEqual(observed, ["response.output_text.delta"]);
      events.push(event);
    }
    assert.equal((await result.result()).stopReason, "error");
    assert.match((await result.result()).errorMessage, /WebSocket closed mid-request/);
    assert.equal(events.filter((event) => event.type === "error").length, 1);
    assert.equal(events.filter((event) => event.type === "done").length, 0);
  });
});

test("reused WebSocket connections report the handshake to each request's observer", { timeout: 5000 }, async () => {
  await withServer(async ({ server, model, context, stream, options }) => {
    let connections = 0;
    let requests = 0;
    server.on("connection", (socket) => {
      connections += 1;
      socket.on("message", () => socket.send(JSON.stringify(completed(`response-${++requests}`))));
    });
    const observations = [];
    for (let request = 0; request < 2; request += 1) {
      const result = stream(model, context, { ...options, onResponse(response) {
        observations.push([request, response.status, response.headers["x-test-observation"]]);
      } });
      assert.equal((await result.result()).stopReason, "stop");
    }
    assert.equal(connections, 1);
    assert.deepEqual(observations, [[0, 101, "present"], [1, 101, "present"]]);
  });
});

test("aborting during an asynchronous observer discards queued provider events", { timeout: 5000 }, async () => {
  await withServer(async ({ server, model, context, stream, options }) => {
    server.on("connection", (socket) => socket.on("message", () => {
      socket.send(JSON.stringify({ type: "response.output_text.delta", delta: "Hello" }));
      socket.send(JSON.stringify(completed()));
    }));
    const controller = new AbortController();
    const observed = [];
    let release;
    const blocked = new Promise((resolve) => { release = resolve; });
    const result = stream(model, context, { ...options, signal: controller.signal,
      async onProviderStreamEvent(event) {
        observed.push(event.type);
        controller.abort();
        await blocked;
      },
    });
    const events = [];
    try {
      for await (const event of result) events.push(event);
      assert.equal((await result.result()).stopReason, "aborted");
    } finally {
      release();
    }
    await delay(10);
    assert.deepEqual(observed, ["response.output_text.delta"]);
    assert.equal(events.filter((event) => event.type === "error").length, 1);
    assert.equal(events.filter((event) => event.type === "done" || event.type === "text_delta").length, 0);
  });
});
