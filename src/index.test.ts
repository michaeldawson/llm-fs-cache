import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  wrapOpenAIWithCache,
  llmCacheKey,
  firebaseEmulatorCacheEnabled,
  firestoreIdNormalizer,
} from "./index";

function makeFakeClient(dir: string) {
  const calls = { responses: 0, chat: 0 };
  const client = wrapOpenAIWithCache(
    {
      responses: {
        create: async (_body: any) => {
          calls.responses += 1;
          return { id: `resp-${calls.responses}`, output_text: "ok" };
        },
      },
      chat: {
        completions: {
          create: async (_body: any) => {
            calls.chat += 1;
            return { id: `chat-${calls.chat}` };
          },
        },
      },
    } as any,
    { dir, enabled: true },
  );
  return { client, calls };
}

function tmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "llm-fs-cache-"));
}

test("key is stable regardless of object key order", () => {
  assert.equal(
    llmCacheKey("responses", { a: 1, b: 2 }),
    llmCacheKey("responses", { b: 2, a: 1 }),
  );
});

test("key separates by surface and changes with the body", () => {
  assert.notEqual(llmCacheKey("responses", { a: 1 }), llmCacheKey("chat.completions", { a: 1 }));
  assert.notEqual(llmCacheKey("responses", { a: 1 }), llmCacheKey("responses", { a: 2 }));
});

test("responses: miss calls through, hit replays, and a file is written", async () => {
  const dir = tmp();
  const { client, calls } = makeFakeClient(dir);
  const body = { model: "gpt", input: "hello" };

  const first = await client.responses.create(body);
  assert.equal(calls.responses, 1);

  const second = await client.responses.create(body);
  assert.equal(calls.responses, 1); // replayed, not called again
  assert.deepEqual(second, first);

  assert.ok(fs.readdirSync(dir).some((f) => f.endsWith(".json")));
});

test("chat.completions is cached too; different body is a miss", async () => {
  const dir = tmp();
  const { client, calls } = makeFakeClient(dir);
  await client.chat.completions.create({ model: "gpt", messages: [{ role: "user", content: "a" }] });
  await client.chat.completions.create({ model: "gpt", messages: [{ role: "user", content: "a" }] });
  assert.equal(calls.chat, 1);
  await client.chat.completions.create({ model: "gpt", messages: [{ role: "user", content: "b" }] });
  assert.equal(calls.chat, 2);
});

test("disabled = pure pass-through, no files", async () => {
  const dir = tmp();
  const calls = { n: 0 };
  const client = wrapOpenAIWithCache(
    { responses: { create: async () => ({ n: ++calls.n }) } } as any,
    { dir, enabled: false },
  );
  await client.responses.create({ model: "gpt", input: "x" });
  await client.responses.create({ model: "gpt", input: "x" });
  assert.equal(calls.n, 2);
  assert.ok(!fs.existsSync(dir) || fs.readdirSync(dir).length === 0);
});

test("streaming requests pass through uncached", async () => {
  const dir = tmp();
  const { client, calls } = makeFakeClient(dir);
  await client.responses.create({ model: "gpt", input: "x", stream: true });
  await client.responses.create({ model: "gpt", input: "x", stream: true });
  assert.equal(calls.responses, 2);
});

test("firestoreIdNormalizer: bodies differing only by a 20-char Firestore id share a key", () => {
  const a = { input: 'tactic {"logId":"6l2dKWUXToF4FBwu3l04"}' };
  const b = { input: 'tactic {"logId":"jx1CuAiBO66M8jouC97f"}' };
  assert.notEqual(llmCacheKey("responses", a), llmCacheKey("responses", b));
  assert.equal(
    llmCacheKey("responses", a, firestoreIdNormalizer),
    llmCacheKey("responses", b, firestoreIdNormalizer),
  );
});

test("firestoreIdNormalizer: leaves short/hyphenated seed ids and prose alone", () => {
  // seeded ids like "sm-log-1" and normal words are untouched (not 20 alnum chars)
  const s = 'behavior [logId=sm-log-1] the quick brown fox jumps over';
  assert.equal(firestoreIdNormalizer(s), s);
});

test("wrapOpenAIWithCache honors keyNormalizer (Firestore ids ignored)", async () => {
  const dir = tmp();
  const calls = { n: 0 };
  const client = wrapOpenAIWithCache(
    { responses: { create: async () => ({ n: ++calls.n }) } } as any,
    { dir, enabled: true, keyNormalizer: firestoreIdNormalizer },
  );
  await client.responses.create({ input: 'x {"logId":"6l2dKWUXToF4FBwu3l04"}' });
  await client.responses.create({ input: 'x {"logId":"jx1CuAiBO66M8jouC97f"}' });
  assert.equal(calls.n, 1); // second is a hit despite the different id
});

test("firebaseEmulatorCacheEnabled: off deployed, on in emulator", () => {
  const save = { ...process.env };
  for (const k of ["K_SERVICE", "FUNCTIONS_EMULATOR", "FIRESTORE_EMULATOR_HOST", "FIREBASE_EMULATOR_HUB", "LLM_CACHE", "NODE_ENV"]) {
    delete process.env[k];
  }
  process.env.K_SERVICE = "svc"; // real deploy, no emulator
  assert.equal(firebaseEmulatorCacheEnabled(), false);
  process.env.FUNCTIONS_EMULATOR = "true"; // functions emulator (also sets K_SERVICE)
  assert.equal(firebaseEmulatorCacheEnabled(), true);
  process.env = save;
});

// A scenario that creates docs mid-conversation: the recording run got ids
// A…/B…, the replay run gets fresh ids X…/Y…, and the model's recorded reply
// cites the recording's id back into a tool call.
const REC_A = "AAAAAAAAAAAAAAAAAAA1";
const REC_B = "BBBBBBBBBBBBBBBBBBB2";
const LIVE_A = "XXXXXXXXXXXXXXXXXXX1";
const LIVE_B = "YYYYYYYYYYYYYYYYYYY2";

function recordFixture(dir: string, body: unknown, response: unknown) {
  const recorder = wrapOpenAIWithCache(
    { responses: { create: async () => response } } as any,
    { dir, enabled: true, keyNormalizer: firestoreIdNormalizer },
  );
  return recorder.responses.create(body);
}

test("remapIds: a replayed response cites the live ids the request carried", async () => {
  const dir = tmp();
  await recordFixture(
    dir,
    { input: `created ${REC_A} and ${REC_B}` },
    { output: [{ arguments: `{"tacticIds":["${REC_B}","${REC_A}"]}` }] },
  );

  const replayer = wrapOpenAIWithCache(
    { responses: { create: async () => assert.fail("should replay") } } as any,
    { dir, enabled: true, keyNormalizer: firestoreIdNormalizer, remapIds: true },
  );
  const replayed = await replayer.responses.create({
    input: `created ${LIVE_A} and ${LIVE_B}`,
  });
  assert.equal(
    replayed.output[0].arguments,
    `{"tacticIds":["${LIVE_B}","${LIVE_A}"]}`,
  );
});

test("remapIds: the mapping carries across calls, and unknown ids are left alone", async () => {
  const dir = tmp();
  const INVENTED = "ZZZZZZZZZZZZZZZZZZZ9";
  // Turn 1 hands the model an id; turn 2's reply cites it (and one it made up)
  // without the id appearing in turn 2's own request.
  await recordFixture(dir, { input: `created ${REC_A}` }, { output_text: "ok" });
  await recordFixture(
    dir,
    { input: "now build the plan" },
    { output_text: `plan uses ${REC_A} and ${INVENTED}` },
  );

  const replayer = wrapOpenAIWithCache(
    { responses: { create: async () => assert.fail("should replay") } } as any,
    { dir, enabled: true, keyNormalizer: firestoreIdNormalizer, remapIds: true },
  );
  await replayer.responses.create({ input: `created ${LIVE_A}` });
  const second = await replayer.responses.create({ input: "now build the plan" });
  assert.equal(second.output_text, `plan uses ${LIVE_A} and ${INVENTED}`);
});

test("without remapIds a replay returns the fixture verbatim", async () => {
  const dir = tmp();
  await recordFixture(dir, { input: `created ${REC_A}` }, { output_text: `id ${REC_A}` });
  const replayer = wrapOpenAIWithCache(
    { responses: { create: async () => assert.fail("should replay") } } as any,
    { dir, enabled: true, keyNormalizer: firestoreIdNormalizer },
  );
  const replayed = await replayer.responses.create({ input: `created ${LIVE_A}` });
  assert.equal(replayed.output_text, `id ${REC_A}`);
});
