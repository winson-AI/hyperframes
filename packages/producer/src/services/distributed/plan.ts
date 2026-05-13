/**
 * Activity A of the distributed render pipeline.
 *
 * `plan(projectDir, config, planDir)` composes the existing render stages
 * (compile → probe → extract videos → audio → freeze) into a self-contained
 * `<planDir>/` directory tree that downstream chunk workers consume:
 *
 *     <planDir>/
 *     ├── plan.json
 *     ├── compiled/                # compileForRender output (self-contained)
 *     ├── video-frames/            # per-video JPEG sequences (dereferenced)
 *     ├── audio.aac                # only when composition has audio
 *     └── meta/
 *         ├── composition.json
 *         ├── encoder.json         # LockedRenderConfig
 *         └── chunks.json
 *
 * Pure function over local paths. No networking. Two invocations with the
 * same inputs produce the same `planHash` — adapters use that contract to
 * short-circuit `plan()` on workflow replay.
 *
 * Banned configurations (GPU encode, hardware browser GL, system primary
 * fonts) are rejected at plan time via `planValidation.ts` so chunk workers
 * never have to handle them.
 */

import { existsSync, mkdirSync, renameSync, rmSync } from "node:fs";
import { join } from "node:path";
import { type CanvasResolution } from "@hyperframes/core";
import { type EngineConfig, resolveConfig } from "@hyperframes/engine";
import { defaultLogger, type ProducerLogger } from "../../logger.js";
import { runAudioStage } from "../render/stages/audioStage.js";
import { runCompileStage } from "../render/stages/compileStage.js";
import { runExtractVideosStage } from "../render/stages/extractVideosStage.js";
import { runProbeStage } from "../render/stages/probeStage.js";
import {
  type ChunkSliceJson,
  type CompositionMetadataJson,
  freezePlan,
  type LockedRenderConfig,
} from "../render/stages/freezePlan.js";
import {
  canonicalJsonStringify,
  type PlanDimensions,
  sha256Hex,
} from "../render/stages/planHash.js";
import { validateNoGpuEncode, validateNoSystemFonts } from "../render/planValidation.js";
import { snapshotRuntimeEnv } from "../render/runtimeEnvSnapshot.js";
import { buildSyntheticRenderJob, readFfmpegVersion, readProducerVersion } from "./shared.js";

/**
 * Caller-supplied configuration for a distributed render. `fps`, `width`,
 * `height`, and `format` are required; everything else carries a default
 * sensible for AWS Lambda fan-out.
 */
export interface DistributedRenderConfig {
  /** Integer frame rate. Distributed renders only accept integer fps; the in-process renderer's `Fps` rational handles NTSC. */
  fps: 24 | 30 | 60;
  width: number;
  height: number;
  /**
   * Output container format. webm and HDR mp4 are not supported in
   * distributed mode — `plan()` refuses them up front with a typed
   * `FormatNotSupportedInDistributedError`. The in-process renderer
   * supports both.
   */
  format: "mp4" | "mov" | "png-sequence";
  quality?: "draft" | "standard" | "high";
  /** Constant-rate-factor override; mutually exclusive with `bitrate`. */
  crf?: number;
  /** Target video bitrate (e.g. `"10M"`); mutually exclusive with `crf`. */
  bitrate?: string;
  /** Output resolution preset; engages Chrome `deviceScaleFactor` supersampling. */
  outputResolution?: CanvasResolution;

  /** Default `240` frames (~8s @ 30fps; fits Lambda's 15-min cap). */
  chunkSize?: number;
  /** Default `16`. Caps long renders to fewer-but-longer chunks for operational fairness. */
  maxParallelChunks?: number;
  /** Runtime hint; consumed by future per-runtime budget checks. The current implementation records the value but does not enforce. */
  runtimeCap?: "lambda" | "temporal" | "cloud-run-job" | "k8s-job" | "none";

  /**
   * Reject compositions whose primary font-family resolves to a host-OS /
   * generic family. Default `true` for distributed renders — overriding to
   * `false` is unsupported and exists only as an escape hatch for tests.
   */
  rejectOnSystemFonts?: boolean;
  /**
   * Threaded into the `injectDeterministicFontFaces` font loader. Default
   * `true` — distributed renders must not silently fall back to system fonts.
   */
  failClosedFontFetch?: boolean;

  /** HDR is not supported in distributed mode; `force-hdr` trips a `FormatNotSupportedInDistributedError`. Defaults to `force-sdr`. */
  hdrMode?: "auto" | "force-sdr";

  logger?: ProducerLogger;
  /** Optional engine config override (env vars are not read when provided). */
  producerConfig?: EngineConfig;
  /** Entry HTML file relative to `projectDir`. Defaults to `"index.html"`. */
  entryFile?: string;
  /** Caller-supplied AbortSignal. Threaded through compile / probe / extract / audio stages. */
  abortSignal?: AbortSignal;
}

/**
 * Result of {@link plan}. The `planHash` is the content-addressed identifier
 * that adapters key replay short-circuits off of.
 */
export interface PlanResult {
  planDir: string;
  planHash: string;
  chunkCount: number;
  totalFrames: number;
  fps: 24 | 30 | 60;
  width: number;
  height: number;
  format: "mp4" | "mov" | "png-sequence";
  ffmpegVersion: string;
  producerVersion: string;
}

/** Default chunk size in frames (~8s @ 30fps; fits Lambda's 15-min cap). */
export const DEFAULT_CHUNK_SIZE = 240;
/** Default cap on parallel chunks for operational fairness across renders. */
export const DEFAULT_MAX_PARALLEL_CHUNKS = 16;

/**
 * Compute `(chunkCount, effectiveChunkSize)` from total frames and the
 * caller's chunking knobs:
 *
 *     chunkCount = min(maxParallelChunks, ceil(totalFrames / chunkSize))
 *     effectiveChunkSize = max(configChunkSize, ceil(totalFrames / maxParallelChunks))
 *
 * Long renders auto-rescale to fewer-but-longer chunks rather than
 * fragmenting infinitely. Returned `chunkCount >= 1` (`totalFrames === 0`
 * is rejected upstream); `effectiveChunkSize >= configChunkSize`.
 */
export function resolveChunkPlan(
  totalFrames: number,
  configChunkSize: number,
  maxParallelChunks: number,
): { chunkCount: number; effectiveChunkSize: number } {
  // Integer-only inputs: a fractional `totalFrames` (e.g. 10.5) would
  // otherwise produce a last chunk with non-integer `endFrame`, and the
  // chunk worker's `for (i = startFrame; i < endFrame; i++)` loop would
  // silently truncate.
  assertPositiveInteger("totalFrames", totalFrames);
  assertPositiveInteger("configChunkSize", configChunkSize);
  assertPositiveInteger("maxParallelChunks", maxParallelChunks);
  const naiveCount = Math.ceil(totalFrames / configChunkSize);
  const chunkCount = Math.min(maxParallelChunks, Math.max(1, naiveCount));
  const effectiveChunkSize = Math.max(configChunkSize, Math.ceil(totalFrames / chunkCount));
  return { chunkCount, effectiveChunkSize };
}

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(
      `[plan] resolveChunkPlan: ${name} must be a positive integer (received ${String(value)})`,
    );
  }
}

/**
 * Slice `totalFrames` into `chunkCount` consecutive ranges. Each chunk gets
 * `effectiveChunkSize` frames except the last, which absorbs the remainder
 * so the union is exactly `[0, totalFrames)`. `endFrame` is the EXCLUSIVE
 * upper bound — chunk workers iterate `i in [startFrame, endFrame)`.
 */
export function buildChunkSlices(
  totalFrames: number,
  chunkCount: number,
  effectiveChunkSize: number,
): ChunkSliceJson[] {
  const slices: ChunkSliceJson[] = [];
  for (let i = 0; i < chunkCount; i++) {
    const startFrame = i * effectiveChunkSize;
    const endFrame =
      i === chunkCount - 1 ? totalFrames : Math.min(totalFrames, startFrame + effectiveChunkSize);
    slices.push({ index: i, startFrame, endFrame });
  }
  return slices;
}

/**
 * Hash the deterministic-font bundle that ships inside `@hyperframes/producer`.
 * The compiled HTML already inlines per-family `@font-face` data URIs, so the
 * snapshot SHA exists primarily to detect cross-version font-bundle drift on
 * chunk workers. Mixed into `planHash`.
 *
 * Pulled lazily because the generated module is large and only the
 * distributed pipeline needs it.
 */
async function readFontSnapshotSha(): Promise<string> {
  const module = (await import("../fontData.generated.js")) as {
    EMBEDDED_FONT_DATA?: unknown;
  };
  const data = module.EMBEDDED_FONT_DATA;
  if (!data || typeof data !== "object") {
    throw new Error(
      "[plan] EMBEDDED_FONT_DATA missing from fontData.generated.js — was `bun run build:fonts` run?",
    );
  }
  // Hash a canonical key fingerprint, not the raw font bytes — the bytes are
  // already mixed in through `compositionHtml` (the @font-face data URIs the
  // compiler injects). What we really want to detect here is "the bundle on
  // worker B is a different version of the producer than on controller A",
  // which is fully captured by the sorted family names + per-family byte
  // lengths.
  const dataObj = data as Record<string, unknown>;
  const fingerprint: Record<string, number> = {};
  for (const key of Object.keys(dataObj).sort()) {
    const value = dataObj[key];
    fingerprint[key] =
      typeof value === "string" ? value.length : JSON.stringify(value ?? null).length;
  }
  return sha256Hex(canonicalJsonStringify(fingerprint));
}

/**
 * Build the `LockedRenderConfig` frozen into `meta/encoder.json`.
 * Captures everything chunk workers need to reproduce the controller's
 * encode decisions byte-for-byte. Validated by the chunk worker on boot —
 * the same input here must round-trip to an identical config.
 */
function buildLockedRenderConfig(input: {
  config: DistributedRenderConfig;
  forceScreenshot: boolean;
  deviceScaleFactor: number;
  ffmpegVersion: string;
  effectiveChunkSize: number;
  chunkCount: number;
  runtimeEnv: Record<string, string>;
}): LockedRenderConfig {
  const { config, forceScreenshot, deviceScaleFactor, ffmpegVersion } = input;
  const { encoder, pixelFormat, preset } = FORMAT_ENCODER_TABLE[config.format];
  return {
    captureMode: forceScreenshot ? "screenshot" : "beginframe",
    forceScreenshot,
    deviceScaleFactor,
    useLayeredHdrComposite: false,
    browserGpuMode: "software",
    // Match `LOCKED_WARMUP_TICKS` in `frameCapture.ts` — kept as a literal so
    // a worker that ships a different value will trip `PLAN_HASH_MISMATCH`
    // (the locked config flows into planHash via the canonical JSON).
    warmupTicks: 60,
    encoder,
    quality: config.quality ?? "standard",
    ffmpegVersion,
    preset,
    crf: config.crf,
    bitrate: config.bitrate,
    // GOP === chunkSize so every chunk's first frame is an IDR keyframe and
    // ffmpeg concat-copy round-trips losslessly.
    gopSize: input.effectiveChunkSize,
    closedGop: true,
    forceKeyframes: "n=0",
    pixelFormat,
    chunkSize: input.effectiveChunkSize,
    chunkCount: input.chunkCount,
    runtimeEnv: input.runtimeEnv,
  };
}

/**
 * Per-format encoder + pixel-format + preset triple. Distributed mode is
 * SDR-only: H.264 8-bit for mp4, ProRes 4444 for mov, raw RGBA for
 * png-sequence.
 */
const FORMAT_ENCODER_TABLE: Record<
  DistributedRenderConfig["format"],
  { encoder: LockedRenderConfig["encoder"]; pixelFormat: string; preset: string }
> = {
  mp4: { encoder: "libx264-software", pixelFormat: "yuv420p", preset: "medium" },
  mov: { encoder: "prores-software", pixelFormat: "yuva444p10le", preset: "4444" },
  "png-sequence": { encoder: "png-sequence", pixelFormat: "rgba", preset: "lossless" },
};

/**
 * Activity A of the distributed render pipeline. Produces a self-contained
 * `<planDir>/` from a project + config. See module docstring for the
 * directory layout.
 */
export async function plan(
  projectDir: string,
  config: DistributedRenderConfig,
  planDir: string,
): Promise<PlanResult> {
  // ── Plan-time validation ──
  // Rejections here surface as typed `PlanValidationError`s with non-retryable
  // codes so workflow adapters don't waste retry budget on banned configs.
  validateNoGpuEncode({
    useGpu: false,
    browserGpuMode: "software",
  });

  if (!existsSync(planDir)) mkdirSync(planDir, { recursive: true });

  const log = config.logger ?? defaultLogger;
  const abortSignal = config.abortSignal;
  const assertNotAborted = (): void => {
    if (abortSignal?.aborted) {
      throw new Error("[plan] render_cancelled");
    }
  };
  const cfg: EngineConfig = {
    ...(config.producerConfig ?? resolveConfig()),
    browserGpuMode: "software",
    forceScreenshot: false,
  };

  const job = buildSyntheticRenderJob({
    fps: { num: config.fps, den: 1 },
    quality: config.quality ?? "standard",
    format: config.format,
    crf: config.crf,
    bitrate: config.bitrate,
    outputResolution: config.outputResolution,
    // HDR is banned in distributed mode. force-sdr keeps the
    // extract / encoder paths off the HDR branches entirely.
    hdrMode: config.hdrMode ?? "force-sdr",
    entryFile: config.entryFile ?? "index.html",
    logger: config.logger,
    producerConfig: config.producerConfig,
  });
  const entryFile = config.entryFile ?? "index.html";
  const htmlPath = join(projectDir, entryFile);
  if (!existsSync(htmlPath)) {
    throw new Error(`[plan] entry file not found: ${htmlPath}`);
  }

  const workDir = join(planDir, ".plan-work");
  if (!existsSync(workDir)) mkdirSync(workDir, { recursive: true });
  const compiledDir = join(workDir, "compiled");

  // The compiled directory lives at `<planDir>/compiled/` in the final
  // layout. The stages write under `<planDir>/.plan-work/compiled/`; we
  // move the contents over once the staged work completes.
  const finalCompiledDir = join(planDir, "compiled");

  // mov + png-sequence carry alpha — flip force-screenshot so compileStage
  // takes the alpha-aware capture path (BeginFrame doesn't preserve alpha
  // on Linux headless-shell).
  const needsAlpha = config.format === "png-sequence" || config.format === "mov";

  // ── Compile ──
  const compileResult = await runCompileStage({
    projectDir,
    workDir,
    htmlPath,
    entryFile,
    job,
    cfg,
    needsAlpha,
    log,
    assertNotAborted,
    // Distributed renders fail closed on font-fetch errors so the planDir
    // is content-addressed against deterministic fonts only.
    failClosedFontFetch: config.failClosedFontFetch !== false,
  });
  let compiled = compileResult.compiled;
  const composition = compileResult.composition;
  const { deviceScaleFactor, forceScreenshot } = compileResult;
  // composition.{width,height} are the authored page dimensions. The
  // post-supersample output dims are `compileResult.outputWidth/outputHeight`
  // — chunks render at output dims, but planHash + composition.json record
  // the page dims so cross-machine consistency keys off the composition's
  // own intent rather than a knob the planner could tweak.
  const { width, height } = composition;

  // ── Reject system primary fonts ──
  // Runs against the post-compile HTML (which has @font-face declarations
  // injected) so we evaluate the same surface the chunk worker would render.
  if (config.rejectOnSystemFonts !== false) {
    validateNoSystemFonts(compiled.html);
  }

  // ── Probe ──
  // Browser probe runs only when needed. For statically-resolvable durations
  // this is a near-zero pass.
  const probeResult = await runProbeStage({
    projectDir,
    workDir,
    job,
    cfg,
    log,
    assertNotAborted,
    compiled,
    composition,
    width,
    height,
    needsAlpha,
    deviceScaleFactor,
  });
  compiled = probeResult.compiled;
  job.duration = probeResult.duration;
  job.totalFrames = probeResult.totalFrames;
  const totalFrames = probeResult.totalFrames;
  if (probeResult.fileServer) probeResult.fileServer.close();
  if (probeResult.probeSession) {
    // Close inside a try/catch — leaking a Chrome process here would mask
    // the original plan() result on cancellation paths.
    try {
      const { closeCaptureSession } = await import("@hyperframes/engine");
      await closeCaptureSession(probeResult.probeSession);
    } catch (err) {
      log.warn("[plan] probe session close failed", {
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // ── Extract videos ──
  // `materializeSymlinks: true` recursively copies frames so the planDir is
  // self-contained (symlinks don't survive S3/GCS round-trips).
  const extractResult = await runExtractVideosStage({
    projectDir,
    compiledDir,
    job,
    cfg,
    composition,
    abortSignal,
    assertNotAborted,
    materializeSymlinks: true,
  });
  if (extractResult.frameLookup) extractResult.frameLookup.cleanup();

  // ── Audio ──
  const audioResult = await runAudioStage({
    projectDir,
    workDir,
    compiledDir,
    duration: job.duration,
    audios: composition.audios,
    abortSignal,
    assertNotAborted,
  });

  // Promote staged artifacts from the temp work tree into the final planDir
  // shape. `workDir` is `<planDir>/.plan-work/` — always the same filesystem
  // as `planDir`, so `renameSync` succeeds without copying. Video frames
  // alone can be hundreds of MB; copying once instead of twice (the prior
  // approach left a duplicate under `compiled/__hyperframes_video_frames/`)
  // halves peak disk usage during `plan()`.
  const stagedVideoFrames = join(compiledDir, "__hyperframes_video_frames");
  const videoFramesDst = join(planDir, "video-frames");
  if (existsSync(videoFramesDst)) rmSync(videoFramesDst, { recursive: true, force: true });
  if (existsSync(stagedVideoFrames)) {
    renameSync(stagedVideoFrames, videoFramesDst);
  } else {
    mkdirSync(videoFramesDst, { recursive: true });
  }

  if (existsSync(finalCompiledDir)) rmSync(finalCompiledDir, { recursive: true, force: true });
  renameSync(compiledDir, finalCompiledDir);

  const planAudioPath = join(planDir, "audio.aac");
  if (audioResult.hasAudio && existsSync(audioResult.audioOutputPath)) {
    renameSync(audioResult.audioOutputPath, planAudioPath);
  }

  // ── Chunking decisions + locked config ──
  const configChunkSize = config.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const maxParallel = config.maxParallelChunks ?? DEFAULT_MAX_PARALLEL_CHUNKS;
  const { chunkCount, effectiveChunkSize } = resolveChunkPlan(
    totalFrames,
    configChunkSize,
    maxParallel,
  );
  const chunks = buildChunkSlices(totalFrames, chunkCount, effectiveChunkSize);

  const ffmpegVersion = await readFfmpegVersion();
  const producerVersion = readProducerVersion();
  const fontSnapshotSha = await readFontSnapshotSha();
  const runtimeEnv = snapshotRuntimeEnv();
  const lockedConfig = buildLockedRenderConfig({
    config,
    forceScreenshot,
    deviceScaleFactor,
    ffmpegVersion,
    effectiveChunkSize,
    chunkCount,
    runtimeEnv,
  });

  // ── Freeze the plan ──
  // `freezePlan` writes meta/{composition,encoder,chunks}.json then walks
  // the planDir to compute planHash from the actual bytes the chunk worker
  // will read.
  const compositionJson: CompositionMetadataJson = {
    durationSeconds: job.duration ?? 0,
    width,
    height,
    fps: job.config.fps,
    videoCount: composition.videos.length,
    audioCount: composition.audios.length,
    imageCount: composition.images.length,
  };
  const dimensions: PlanDimensions = {
    fpsNum: config.fps,
    fpsDen: 1,
    width,
    height,
    format: config.format,
  };
  const freezeResult = await freezePlan({
    planDir,
    composition: compositionJson,
    encoder: lockedConfig,
    chunks,
    dimensions,
    producerVersion,
    fontSnapshotSha,
    durationSeconds: job.duration ?? 0,
    totalFrames,
    hasAudio: audioResult.hasAudio,
  });
  const planHash = freezeResult.planHash;

  // Clean up the temp work tree. `.plan-work/` holds intermediate
  // compileStage artifacts that are now promoted into `planDir/`; leaving
  // it would inflate the planDir-size check and confuse chunk workers' file
  // walks.
  try {
    rmSync(workDir, { recursive: true, force: true });
  } catch (err) {
    log.warn("[plan] failed to remove temp work dir", {
      workDir,
      error: err instanceof Error ? err.message : String(err),
    });
  }

  return {
    planDir,
    planHash,
    chunkCount,
    totalFrames,
    fps: config.fps,
    width,
    height,
    format: config.format,
    ffmpegVersion,
    producerVersion,
  };
}
