/**
 * llm-fs-cache — a filesystem-backed read-through cache for LLM SDK clients.
 *
 * Wrap a client once and every matching request is keyed by
 * SHA256(api-surface + request body): a cache HIT replays the stored response
 * from disk, a MISS calls the provider and writes the response as `<hash>.json`.
 * Commit those files and your LLM-backed tests replay for free in CI instead of
 * hitting the paid API.
 *
 * OpenAI is supported today (`wrapOpenAIWithCache`); the store, keying and
 * gating are provider-agnostic, so other providers can be added the same way.
 *
 * There are no streaming call sites to reassemble here — non-streaming requests
 * return a full response object, so caching the whole thing is sufficient. (A
 * streaming call is passed straight through, never cached.)
 */
import { createHash } from "crypto";
import * as fs from "fs";
import * as path from "path";

export interface CacheOptions {
  /**
   * Directory the `<hash>.json` fixtures live in. Default:
   * `process.env.LLM_CACHE_DIR` or `<cwd>/.llm-cache`.
   */
  dir?: string;
  /**
   * Whether the cache intercepts calls. A boolean, or a function evaluated per
   * call (so env/context changes take effect). Default: `defaultCacheEnabled()`
   * — opt-in via `LLM_CACHE=1`. Firebase apps can pass
   * `firebaseEmulatorCacheEnabled` to auto-enable against emulators only.
   */
  enabled?: boolean | (() => boolean);
  /**
   * Transform the serialized request before it's hashed into the cache key. Use
   * this to blank out volatile-but-irrelevant values (e.g. random Firestore doc
   * ids) so two requests that differ only in those still share a key. The full
   * body is still sent and stored — only the key ignores the blanked bits.
   * Ships `firestoreIdNormalizer` for the common Firebase case.
   */
  keyNormalizer?: (serialized: string) => string;
  /**
   * Rewrite ids in replayed responses from the recording run's values to this
   * run's. For requests whose ids `keyNormalizer` blanks but the model DOES
   * depend on: a tool result hands the model a freshly created doc id, and the
   * model's next call cites it. On replay those docs have new ids, so the
   * fixture's reply would cite ids that no longer exist.
   *
   * On every hit, the stored request and the live one normalize to the same
   * key, so their ids pair up by position; each recorded→live pair is kept for
   * the life of the wrapped client and applied to every replayed response. Ids
   * never seen in a request (e.g. ones the model invented) are left alone.
   *
   * `true` uses the Firestore auto-id pattern (pair it with
   * `firestoreIdNormalizer`); pass a global RegExp for other id shapes — it must
   * match exactly the ids your `keyNormalizer` blanks.
   */
  remapIds?: boolean | RegExp;
}

/**
 * Blanks standalone 20-char Firestore auto-ids (`[A-Za-z0-9]{20}`) so cache keys
 * are stable when a request embeds random doc ids the model doesn't depend on
 * (e.g. a `logId` replayed in tool-call results). Pass as `keyNormalizer`.
 */
const FIRESTORE_ID_PATTERN = /(?<![A-Za-z0-9])[A-Za-z0-9]{20}(?![A-Za-z0-9])/g;

export function firestoreIdNormalizer(serialized: string): string {
  return serialized.replace(FIRESTORE_ID_PATTERN, "<id>");
}

/** Opt-in default: on only when `LLM_CACHE` is truthy. */
export function defaultCacheEnabled(): boolean {
  const flag =
    typeof process !== "undefined"
      ? process.env?.LLM_CACHE?.toLowerCase()
      : undefined;
  return flag === "1" || flag === "true" || flag === "on";
}

/**
 * Enablement helper for Firebase apps: ON against the emulators (tests/local),
 * hard-OFF in real deployed Cloud Functions / Cloud Run — a production cache
 * would replay identical model output to real users. `LLM_CACHE=0` forces off.
 */
export function firebaseEmulatorCacheEnabled(): boolean {
  const env = typeof process !== "undefined" ? process.env ?? {} : {};
  const inEmulator = !!(
    env.FUNCTIONS_EMULATOR ||
    env.FIRESTORE_EMULATOR_HOST ||
    env.FIREBASE_EMULATOR_HUB
  );
  // Real deployed functions set K_SERVICE and are NOT an emulator — hard off
  // there, even over an explicit LLM_CACHE=1 (the functions emulator also sets
  // K_SERVICE, so the emulator check is what keeps it active in tests).
  if (env.K_SERVICE && !inEmulator) return false;
  const flag = env.LLM_CACHE?.toLowerCase();
  if (flag === "0" || flag === "false" || flag === "off") return false;
  if (flag === "1" || flag === "true" || flag === "on") return true;
  return inEmulator && env.NODE_ENV !== "production";
}

/** Deterministic JSON: keys sorted recursively so key order can't change the hash. */
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) return "[" + value.map(stableStringify).join(",") + "]";
  const obj = value as Record<string, unknown>;
  return (
    "{" +
    Object.keys(obj)
      .sort()
      .filter((k) => obj[k] !== undefined)
      .map((k) => JSON.stringify(k) + ":" + stableStringify(obj[k]))
      .join(",") +
    "}"
  );
}

/** SHA256 over the API surface + normalized request body. */
export function llmCacheKey(
  surface: string,
  body: unknown,
  keyNormalizer?: (serialized: string) => string,
): string {
  let serialized = stableStringify(body);
  if (keyNormalizer) serialized = keyNormalizer(serialized);
  return createHash("sha256").update(surface + "\n" + serialized).digest("hex");
}

function resolveDir(opts: CacheOptions): string {
  if (opts.dir) return opts.dir;
  const env = typeof process !== "undefined" ? process.env?.LLM_CACHE_DIR : undefined;
  return env || path.join(process.cwd(), ".llm-cache");
}

function isEnabled(opts: CacheOptions): boolean {
  if (typeof opts.enabled === "function") return opts.enabled();
  if (typeof opts.enabled === "boolean") return opts.enabled;
  return defaultCacheEnabled();
}

function readCache(
  dir: string,
  key: string,
): { request: unknown; response: unknown } | undefined {
  const file = path.join(dir, key + ".json");
  if (!fs.existsSync(file)) return undefined;
  try {
    const stored = JSON.parse(fs.readFileSync(file, "utf8"));
    if (stored.response === undefined) return undefined;
    return { request: stored.request, response: stored.response };
  } catch {
    return undefined;
  }
}

function idPattern(remapIds: true | RegExp): RegExp {
  const source = remapIds === true ? FIRESTORE_ID_PATTERN : remapIds;
  return new RegExp(source.source, source.flags.includes("g") ? source.flags : source.flags + "g");
}

/** Pair the recording's ids with this run's, by position, into `idMap`. */
function learnIds(
  pattern: RegExp,
  recordedRequest: unknown,
  liveRequest: unknown,
  idMap: Map<string, string>,
): void {
  const recorded = stableStringify(recordedRequest).match(pattern) ?? [];
  const live = stableStringify(liveRequest).match(pattern) ?? [];
  // Same key ⇒ same skeleton, so the counts agree unless the pattern and the
  // keyNormalizer disagree about what an id is; then pairing would be a guess.
  if (recorded.length !== live.length) return;
  recorded.forEach((id, i) => {
    if (id !== live[i]) idMap.set(id, live[i]);
  });
}

function remapResponse(
  pattern: RegExp,
  response: unknown,
  idMap: Map<string, string>,
): unknown {
  if (idMap.size === 0) return response;
  const serialized = JSON.stringify(response);
  const remapped = serialized.replace(pattern, (id) => idMap.get(id) ?? id);
  return remapped === serialized ? response : JSON.parse(remapped);
}

function writeCache(
  dir: string,
  key: string,
  surface: string,
  request: unknown,
  response: unknown,
): void {
  fs.mkdirSync(dir, { recursive: true });
  const model =
    request && typeof request === "object"
      ? (request as { model?: string }).model
      : undefined;
  // `request` is stored for reviewability (a new fixture in a PR = a prompt
  // changed); `response` is what's replayed. No timestamps, so re-recording an
  // unchanged request produces no diff.
  fs.writeFileSync(
    path.join(dir, key + ".json"),
    JSON.stringify({ surface, model, request, response }, null, 2),
  );
}

function cachedCreate(
  surface: string,
  realCreate: (...args: any[]) => Promise<any>,
  opts: CacheOptions,
  idMap: Map<string, string>,
): (...args: any[]) => Promise<any> {
  return async (body: any, ...rest: any[]) => {
    // Streaming responses can't be replayed from a stored object — pass through.
    if (!isEnabled(opts) || body?.stream) return realCreate(body, ...rest);
    const dir = resolveDir(opts);
    const key = llmCacheKey(surface, body, opts.keyNormalizer);
    const hit = readCache(dir, key);
    if (hit !== undefined) {
      if (!opts.remapIds) return hit.response;
      const pattern = idPattern(opts.remapIds);
      learnIds(pattern, hit.request, body, idMap);
      return remapResponse(pattern, hit.response, idMap);
    }
    const response = await realCreate(body, ...rest);
    writeCache(dir, key, surface, body, response);
    return response;
  };
}

/**
 * Return a proxied OpenAI client whose `responses.create` and
 * `chat.completions.create` are read-through cached. All other properties pass
 * through untouched. Wrap the single client your app funnels calls through and
 * every call site is covered with no other changes.
 *
 * @example
 * import OpenAI from "openai";
 * import { wrapOpenAIWithCache, firebaseEmulatorCacheEnabled } from "llm-fs-cache";
 * const client = wrapOpenAIWithCache(new OpenAI(), {
 *   dir: "test/llm-cache",
 *   enabled: firebaseEmulatorCacheEnabled,
 * });
 */
export function wrapOpenAIWithCache<T extends object>(
  client: T,
  opts: CacheOptions = {},
): T {
  // Recorded→live id pairs (see `remapIds`), shared by every surface of this
  // client for its lifetime: an id learned from one call's request is what a
  // later call's replayed reply cites.
  const idMap = new Map<string, string>();
  const wrapCompletions = (completions: any) =>
    new Proxy(completions, {
      get(t, p, r) {
        if (p === "create") {
          return cachedCreate(
            "chat.completions",
            (...a: any[]) => t.create(...a),
            opts,
            idMap,
          );
        }
        return Reflect.get(t, p, r);
      },
    });

  const wrapChat = (chat: any) =>
    new Proxy(chat, {
      get(t, p, r) {
        if (p === "completions") return wrapCompletions(Reflect.get(t, p, r));
        return Reflect.get(t, p, r);
      },
    });

  const wrapResponses = (responses: any) =>
    new Proxy(responses, {
      get(t, p, r) {
        if (p === "create") {
          return cachedCreate("responses", (...a: any[]) => t.create(...a), opts, idMap);
        }
        return Reflect.get(t, p, r);
      },
    });

  return new Proxy(client, {
    get(t, p, r) {
      if (p === "responses") return wrapResponses(Reflect.get(t, p, r));
      if (p === "chat") return wrapChat(Reflect.get(t, p, r));
      return Reflect.get(t, p, r);
    },
  });
}
