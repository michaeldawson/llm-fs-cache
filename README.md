# llm-fs-cache

A filesystem-backed **read-through cache** for LLM SDK clients. Wrap your client once; every request is keyed by `SHA256(api-surface + request body)`. A cache **hit** replays the stored response from disk; a **miss** calls the provider and writes the response as `<hash>.json`. Commit those fixtures and your LLM-backed tests **replay for free in CI** instead of hitting the paid API.

OpenAI is supported today. The store, keying, and gating are provider-agnostic, so other providers can be added the same way.

## Install

```bash
npm install llm-fs-cache
```

## Use

Wrap the single client your app funnels calls through — no call-site changes:

```ts
import OpenAI from "openai";
import { wrapOpenAIWithCache } from "llm-fs-cache";

const client = wrapOpenAIWithCache(new OpenAI(), {
  dir: "test/llm-cache",   // where <hash>.json fixtures live
  enabled: process.env.CI === "true" || process.env.NODE_ENV === "test",
});

// unchanged — now cached:
await client.responses.create({ model: "gpt-4.1-mini", input: "hi" });
await client.chat.completions.create({ model: "gpt-4.1-mini", messages: [...] });
```

`responses.create` and `chat.completions.create` are cached; everything else passes through. Streaming requests (`stream: true`) are never cached.

## Options

| option | default | meaning |
|---|---|---|
| `dir` | `process.env.LLM_CACHE_DIR` or `<cwd>/.llm-cache` | fixture directory |
| `enabled` | `defaultCacheEnabled()` (`LLM_CACHE=1`) | `boolean` or a `() => boolean` evaluated per call |

### Firebase apps

`firebaseEmulatorCacheEnabled` auto-enables the cache against the Firebase emulators (tests/local) and is **hard-off in deployed Cloud Functions / Cloud Run** — a production cache would replay identical model output to real users:

```ts
import { wrapOpenAIWithCache, firebaseEmulatorCacheEnabled } from "llm-fs-cache";

const client = wrapOpenAIWithCache(new OpenAI(), {
  dir: "test/llm-cache",
  enabled: firebaseEmulatorCacheEnabled,
});
```

(The Firebase functions emulator also sets `K_SERVICE`, so the gate keys off emulator signals — `FUNCTIONS_EMULATOR` / `FIRESTORE_EMULATOR_HOST` — to distinguish it from a real deploy. `LLM_CACHE=0` forces off, `LLM_CACHE=1` forces on except in a real deploy.)

## Fixtures

Each `<hash>.json` is `{ surface, model, request, response }`. `request` is stored for reviewability — a new fixture in a PR means a prompt changed. `response` is what's replayed. No timestamps are written, so re-recording an unchanged request produces no diff. Commit the directory so CI and new devs replay for free.

**Determinism note:** a fixture only reuses across runs if the request body is byte-identical. If your prompts embed wall-clock time or other nondeterministic values, freeze them in the test path (e.g. a `TEST_NOW_MS`-style clock) so the keys stay stable.

## API

- `wrapOpenAIWithCache<T>(client: T, opts?: CacheOptions): T`
- `llmCacheKey(surface: string, body: unknown): string`
- `defaultCacheEnabled(): boolean` — on when `LLM_CACHE` is truthy
- `firebaseEmulatorCacheEnabled(): boolean` — Firebase emulator gate

## License

MIT
