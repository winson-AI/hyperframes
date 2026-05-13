/**
 * Helpers shared between the distributed activity scripts (`plan.ts`,
 * `renderChunk.ts`, `assemble.ts`). Kept module-local so the public surface
 * stays just the three activity functions plus their result types.
 */

import { execFile as execFileCallback } from "node:child_process";
import { dirname, join } from "node:path";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { type Fps } from "@hyperframes/core";
import { type RenderConfig, type RenderJob, createRenderJob } from "../renderOrchestrator.js";
import { defaultLogger, type ProducerLogger } from "../../logger.js";

const execFile = promisify(execFileCallback);

/**
 * Cached first line of `ffmpeg -version` (e.g. `"ffmpeg version 6.1.1"`).
 * Cached because workers that fan out multiple `renderChunk()` calls in the
 * same process (Cloud Run Jobs, Temporal activity workers) would otherwise
 * spawn ffmpeg once per chunk just to read the version — ~20-50ms each.
 */
let cachedFfmpegVersion: string | null = null;

/**
 * Read `ffmpeg -version` first line. The string is opaque — `planHash`
 * mixes it in verbatim, so any drift across worker hosts trips a
 * `FFMPEG_VERSION_MISMATCH` rather than producing pixels that subtly
 * disagree with the plan's baked-in encoder args.
 */
export async function readFfmpegVersion(): Promise<string> {
  if (cachedFfmpegVersion !== null) return cachedFfmpegVersion;
  const { stdout } = await execFile("ffmpeg", ["-version"], { maxBuffer: 1024 * 1024 });
  const firstLine = stdout.split(/\r?\n/)[0]?.trim() ?? "";
  if (!firstLine) {
    throw new Error("ffmpeg -version returned empty output");
  }
  cachedFfmpegVersion = firstLine;
  return firstLine;
}

/** Test-only: clear the cached ffmpeg version so a fresh probe runs. */
export function _resetFfmpegVersionCacheForTests(): void {
  cachedFfmpegVersion = null;
}

/**
 * Inputs for {@link buildSyntheticRenderJob}. The two distributed activity
 * scripts (`plan.ts`, `renderChunk.ts`) reach for slightly different
 * sources — caller config vs. frozen `LockedRenderConfig` — but the
 * resulting `RenderJob` shape is identical, so the helper accepts both.
 */
export interface SyntheticRenderJobInput {
  fps: Fps;
  format: RenderConfig["format"];
  quality: RenderConfig["quality"];
  crf?: number;
  bitrate?: string;
  outputResolution?: RenderConfig["outputResolution"];
  hdrMode: RenderConfig["hdrMode"];
  entryFile: string;
  logger?: ProducerLogger;
  producerConfig?: RenderConfig["producerConfig"];
}

/**
 * Synthesize a `RenderJob` from a distributed-render config. The distributed
 * activities operate without a full `RenderJob` (they're stateless workers),
 * so we build one to feed the existing stage interfaces.
 */
export function buildSyntheticRenderJob(input: SyntheticRenderJobInput): RenderJob {
  const renderConfig: RenderConfig = {
    fps: input.fps,
    quality: input.quality,
    format: input.format,
    crf: input.crf,
    videoBitrate: input.bitrate,
    outputResolution: input.outputResolution,
    // Distributed mode hard-pins to software GPU. The plan-time validator
    // refuses to fan out otherwise.
    useGpu: false,
    debug: false,
    entryFile: input.entryFile,
    logger: input.logger ?? defaultLogger,
    hdrMode: input.hdrMode,
    producerConfig: input.producerConfig,
  };
  return createRenderJob(renderConfig);
}

/**
 * Resolve the producer package version by walking up from the calling
 * module until a `package.json` whose `name === "@hyperframes/producer"`
 * is found. Works for both the bundled `dist/index.js` (1 level up) and
 * the unbundled source tree (4 levels up).
 *
 * Cached at module load — the version is fixed for the life of the process,
 * and reading the package.json over and over wastes per-chunk syscalls.
 */
let cachedProducerVersion: string | null = null;
export function readProducerVersion(): string {
  if (cachedProducerVersion !== null) return cachedProducerVersion;
  const startDir = dirname(fileURLToPath(import.meta.url));
  let current = startDir;
  for (let i = 0; i < 10; i++) {
    const candidate = join(current, "package.json");
    if (existsSync(candidate)) {
      try {
        const pkg = JSON.parse(readFileSync(candidate, "utf-8")) as {
          name?: string;
          version?: string;
        };
        if (pkg.name === "@hyperframes/producer" && typeof pkg.version === "string") {
          cachedProducerVersion = pkg.version;
          return pkg.version;
        }
      } catch {
        // Fall through to the next ancestor.
      }
    }
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  cachedProducerVersion = "0.0.0-unknown";
  return cachedProducerVersion;
}
