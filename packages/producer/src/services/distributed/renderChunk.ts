/**
 * Activity B of the distributed render pipeline.
 *
 * `renderChunk(planDir, chunkIndex, outputChunkPath)` validates the planDir
 * against the worker's environment, captures the chunk's frame range, and
 * encodes a single closed-GOP video chunk (or, for png-sequence, a directory
 * of PNGs). The output is byte-identical across retries on the same worker
 * and PSNR-equivalent across workers — that contract is what makes Temporal
 * activity retries safe.
 *
 * Pure function over local paths. No networking. Spins up its own headless
 * Chrome + file server scoped to the chunk; tears them down before
 * returning. The caller is responsible for moving `outputChunkPath` to its
 * orchestration-level storage (S3 / GCS / EFS / …).
 *
 * Hard contracts:
 *   - The worker re-applies `meta/encoder.json.runtimeEnv` into
 *     `process.env` BEFORE the file server starts so the served HTML's
 *     `RENDER_MODE_SCRIPT` sees the same env it would have seen on the
 *     controller.
 *   - Browser is launched with `browserGpuMode: "software"` and verified
 *     against `chrome://gpu` via `assertSwiftShader` — a non-SwiftShader
 *     backend trips a non-retryable `BROWSER_GPU_NOT_SOFTWARE`.
 *   - The file server serves with the seeded-random shim
 *     (`buildVirtualTimeShim({ seedRandomFromFrame: true })`) so any
 *     composition that uses `Math.random` / `crypto.getRandomValues`
 *     produces byte-identical pixels per `(planDir, chunkIndex)`.
 *   - One `discardWarmupCapture` runs before the chunk's first real frame
 *     to prime the BeginFrame `lastFrameCache`.
 *   - The chunk's encode runs with `lockGopForChunkConcat: true` and
 *     `gopSize === framesInChunk` so concat-copy at assemble time is safe.
 *
 * Every determinism toggle above is opt-in — only this primitive enables them.
 * In-process renders (`executeRenderJob`) leave them off.
 */

import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Page } from "puppeteer-core";
import {
  assertSwiftShader,
  BROWSER_GPU_NOT_SOFTWARE,
  type CaptureOptions,
  type CaptureSession,
  closeCaptureSession,
  createCaptureSession,
  discardWarmupCapture,
  type EngineConfig,
  getEncoderPreset,
  initializeSession,
  resolveConfig,
} from "@hyperframes/engine";
import { defaultLogger } from "../../logger.js";
import { runEncodeStage } from "../render/stages/encodeStage.js";
import { runCaptureStage } from "../render/stages/captureStage.js";
import {
  type ChunkSliceJson,
  type LockedRenderConfig,
  recomputePlanHashFromPlanDir,
} from "../render/stages/freezePlan.js";
import { sha256Hex } from "../render/stages/planHash.js";
import { applyRuntimeEnvSnapshot } from "../render/runtimeEnvSnapshot.js";
import { buildVirtualTimeShim, createFileServer, type FileServerHandle } from "../fileServer.js";
import { buildSyntheticRenderJob, readFfmpegVersion } from "./shared.js";

/**
 * Non-retryable error codes raised when the planDir is structurally
 * malformed, semantically out of range, or fingerprints differently from
 * what the controller wrote. Each is distinct so adapter retry policies
 * can route them independently — e.g. `MISSING_PLAN_ARTIFACT` may point
 * to a partial S3 download that a retry could heal, while
 * `PLAN_HASH_MISMATCH` strictly indicates cross-version drift that
 * retries won't fix.
 */
export const FFMPEG_VERSION_MISMATCH = "FFMPEG_VERSION_MISMATCH";
export const PLAN_HASH_MISMATCH = "PLAN_HASH_MISMATCH";
export const MISSING_PLAN_ARTIFACT = "MISSING_PLAN_ARTIFACT";
export const CHUNK_INDEX_OUT_OF_RANGE = "CHUNK_INDEX_OUT_OF_RANGE";
export const MISSING_RUNTIME_ENV_SNAPSHOT = "MISSING_RUNTIME_ENV_SNAPSHOT";

export type RenderChunkValidationCode =
  | typeof FFMPEG_VERSION_MISMATCH
  | typeof PLAN_HASH_MISMATCH
  | typeof MISSING_PLAN_ARTIFACT
  | typeof CHUNK_INDEX_OUT_OF_RANGE
  | typeof MISSING_RUNTIME_ENV_SNAPSHOT
  | typeof BROWSER_GPU_NOT_SOFTWARE;

/**
 * Typed non-retryable error raised by `renderChunk` when the planDir is
 * malformed or the worker's runtime doesn't match the planDir's
 * controller-side fingerprint. Workflow adapters key retry policies off
 * `code` — most of these failures will not heal on retry.
 */
export class RenderChunkValidationError extends Error {
  readonly code: RenderChunkValidationCode;
  constructor(code: RenderChunkValidationCode, message: string) {
    super(message);
    this.name = "RenderChunkValidationError";
    this.code = code;
  }
}

/**
 * Result of {@link renderChunk}. The `sha256` field is the byte hash of the
 * primary output (the mp4/mov file, or, for png-sequence, the sorted-frame
 * fingerprint). Retries on the same `(planDir, chunkIndex)` MUST produce
 * the same `sha256` — that contract is the byte-identical-retry axis.
 */
export interface ChunkResult {
  /** Absolute path the encoded chunk was written to (file or directory). */
  outputPath: string;
  /** `"file"` for mp4/mov; `"frame-dir"` for png-sequence. */
  outputKind: "file" | "frame-dir";
  framesEncoded: number;
  sha256: string;
  durationMs: number;
  /**
   * Path to a sidecar JSON containing per-chunk perf counters. Adapters
   * upload this alongside the chunk so per-chunk regressions are
   * inspectable without the workflow having to carry the payload.
   */
  perfPath: string;
}

/** Plan-time JSON manifest written by `freezePlan`. */
interface PlanJson {
  planHash: string;
  producerVersion: string;
  ffmpegVersion: string;
  fontSnapshotSha: string;
  dimensions: {
    fpsNum: number;
    fpsDen: number;
    width: number;
    height: number;
    format: "mp4" | "mov" | "png-sequence" | "webm";
  };
  chunkCount: number;
  totalFrames: number;
  duration: number;
  hasAudio: boolean;
}

/**
 * Re-export the runtime-env apply helper so adapters that import only
 * this subpath can prime `process.env` before instantiating their own
 * file server. Returns a `{ restore }` handle — adapters that fan out
 * multiple chunks per process MUST call `restore()` between chunks.
 */
export { applyRuntimeEnvSnapshot } from "../render/runtimeEnvSnapshot.js";

/**
 * Read SwiftShader vendor/renderer via a 1×1 WebGL canvas + the
 * `WEBGL_debug_renderer_info` extension. Used as the `readInfo` override
 * for {@link assertSwiftShader} when the worker is running on
 * `chrome-headless-shell` — that build serves `chrome://gpu` as an empty
 * document so the default `chrome://gpu`-based info reader trips
 * `net::ERR_FAILED` even when the GL backend is in fact SwiftShader.
 *
 * The canvas-based probe runs against whatever page the caller hands in
 * (we use a fresh `about:blank` so it doesn't depend on the composition
 * URL being navigated yet). The renderer string returned matches the
 * format `assertSwiftShader` expects (substring match against
 * `"swiftshader"`).
 */
export async function readWebGlVendorInfoFromCanvas(
  page: Page,
): Promise<{ vendor: string; renderer: string }> {
  await page.goto("about:blank", { waitUntil: "domcontentloaded", timeout: 30_000 });
  return page.evaluate((): { vendor: string; renderer: string } => {
    try {
      const canvas = document.createElement("canvas");
      const gl =
        (canvas.getContext("webgl") as WebGLRenderingContext | null) ??
        (canvas.getContext("experimental-webgl") as WebGLRenderingContext | null);
      if (!gl) {
        return { vendor: "", renderer: "" };
      }
      const ext = gl.getExtension("WEBGL_debug_renderer_info") as {
        UNMASKED_VENDOR_WEBGL: number;
        UNMASKED_RENDERER_WEBGL: number;
      } | null;
      if (!ext) {
        return {
          vendor: String(gl.getParameter(gl.VENDOR) ?? ""),
          renderer: String(gl.getParameter(gl.RENDERER) ?? ""),
        };
      }
      // Older Chrome builds expose the unmasked strings under the literal
      // numeric constants 0x9245 / 0x9246. The extension surface above is
      // identical across builds — read through it.
      return {
        vendor: String(gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) ?? ""),
        renderer: String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) ?? ""),
      };
    } catch {
      return { vendor: "", renderer: "" };
    }
  });
}

/**
 * Compute a deterministic SHA-256 fingerprint for the chunk's output.
 *
 *   - file output (mp4/mov): straight hash of the file bytes.
 *   - frame-dir (png-sequence): hash the sorted list of `(name, sha256)`
 *     pairs. Avoids the cost of streaming every frame's contents through
 *     a single sha context while still detecting any byte-level drift in
 *     any individual frame.
 *
 * The fingerprint flows into the `ChunkResult.sha256` which adapters
 * compare across retries to enforce the byte-identical-retry contract.
 */
function hashChunkOutput(outputPath: string, kind: "file" | "frame-dir"): string {
  if (kind === "file") return sha256Hex(readFileSync(outputPath));
  const entries = readdirSync(outputPath)
    .filter((name) => /\.(png|jpg|jpeg)$/i.test(name))
    .sort();
  // Hash the sorted (name, perFileSha) list. Encoded as null-separated
  // utf-8 to keep concatenation unambiguous if a frame name ever contains
  // an unusual character.
  const lines = entries.map(
    (name) => `${name}\0${sha256Hex(readFileSync(join(outputPath, name)))}`,
  );
  return sha256Hex(lines.join("\0"));
}

/**
 * Activity B: render a single chunk of the planDir. The `outputChunkPath`
 * argument is a file for mp4/mov outputs and a directory for png-sequence
 * outputs — the caller picks the right shape based on `meta/encoder.json`.
 * `renderChunk` enforces the same choice via `outputKind` on the result.
 */
export async function renderChunk(
  planDir: string,
  chunkIndex: number,
  outputChunkPath: string,
): Promise<ChunkResult> {
  const start = Date.now();
  const log = defaultLogger;

  // ── Read + validate the plan ──
  const planJsonPath = join(planDir, "plan.json");
  const encoderJsonPath = join(planDir, "meta", "encoder.json");
  const chunksJsonPath = join(planDir, "meta", "chunks.json");
  for (const required of [planJsonPath, encoderJsonPath, chunksJsonPath]) {
    if (!existsSync(required)) {
      throw new RenderChunkValidationError(
        MISSING_PLAN_ARTIFACT,
        `[renderChunk] planDir is missing required artifact: ${required}`,
      );
    }
  }
  const plan = JSON.parse(readFileSync(planJsonPath, "utf-8")) as PlanJson;
  const encoder = JSON.parse(readFileSync(encoderJsonPath, "utf-8")) as LockedRenderConfig;
  const chunks = JSON.parse(readFileSync(chunksJsonPath, "utf-8")) as ChunkSliceJson[];

  if (chunkIndex < 0 || chunkIndex >= chunks.length) {
    throw new RenderChunkValidationError(
      CHUNK_INDEX_OUT_OF_RANGE,
      `[renderChunk] chunkIndex ${chunkIndex} is out of range [0, ${chunks.length})`,
    );
  }
  // The bounds check above guarantees this hits, but TS doesn't narrow
  // the indexed access — re-check explicitly.
  const slice = chunks[chunkIndex];
  if (slice === undefined) {
    throw new RenderChunkValidationError(CHUNK_INDEX_OUT_OF_RANGE, "[renderChunk] missing slice");
  }
  const framesInChunk = slice.endFrame - slice.startFrame;
  if (framesInChunk <= 0) {
    throw new RenderChunkValidationError(
      CHUNK_INDEX_OUT_OF_RANGE,
      `[renderChunk] chunk ${chunkIndex} has non-positive frame count: ${framesInChunk}`,
    );
  }

  const compiledDir = join(planDir, "compiled");
  if (!existsSync(compiledDir)) {
    throw new RenderChunkValidationError(
      MISSING_PLAN_ARTIFACT,
      `[renderChunk] planDir missing compiled/ directory: ${compiledDir}`,
    );
  }

  // ── Cross-version sanity ──
  const ffmpegVersion = await readFfmpegVersion();
  if (ffmpegVersion !== plan.ffmpegVersion) {
    throw new RenderChunkValidationError(
      FFMPEG_VERSION_MISMATCH,
      `[renderChunk] ffmpeg version on this worker does not match planDir. ` +
        `planDir: ${JSON.stringify(plan.ffmpegVersion)}; worker: ${JSON.stringify(ffmpegVersion)}. ` +
        `Distributed retries require byte-identical ffmpeg builds across workers. ` +
        `Re-plan from a worker matching this version, or run all renders on an image with the planDir's ffmpeg.`,
    );
  }
  if (encoder.browserGpuMode !== "software") {
    throw new RenderChunkValidationError(
      BROWSER_GPU_NOT_SOFTWARE,
      `[renderChunk] planDir requires browserGpuMode=software, got ${JSON.stringify(encoder.browserGpuMode)}.`,
    );
  }

  // Re-derive `planHash` from the on-disk bytes and compare to the value
  // the controller wrote into `plan.json`. Catches corrupted artifacts
  // (truncated meta files, partial S3 downloads, manual tampering) before
  // the chunk renders. Distinct from the other validation paths above
  // because `MISSING_PLAN_ARTIFACT` etc. are structural; this is purely
  // content-fingerprint drift.
  const recomputedPlanHash = recomputePlanHashFromPlanDir(planDir);
  if (recomputedPlanHash !== plan.planHash) {
    throw new RenderChunkValidationError(
      PLAN_HASH_MISMATCH,
      `[renderChunk] planDir content fingerprint does not match plan.json.planHash. ` +
        `plan.json: ${plan.planHash}; recomputed: ${recomputedPlanHash}. ` +
        `Likely a corrupted artifact (partial S3 download, manual tampering) or a planDir ` +
        `produced by an incompatible producer version. Re-plan and re-fan-out.`,
    );
  }

  // Distinct from the silent `?? {}` fallback we used before: missing
  // `runtimeEnv` means the planDir was produced by a controller that
  // forgot to snapshot, and the chunk's pixels would diverge silently.
  // Surface it as a typed validation error so the workflow can re-plan.
  if (!encoder.runtimeEnv || typeof encoder.runtimeEnv !== "object") {
    throw new RenderChunkValidationError(
      MISSING_RUNTIME_ENV_SNAPSHOT,
      "[renderChunk] planDir is missing meta/encoder.json.runtimeEnv snapshot. " +
        "Re-plan with the current producer.",
    );
  }

  // Apply the controller's runtime-env snapshot. Must happen BEFORE the
  // file server is created — RENDER_MODE_SCRIPT bakes env vars into
  // served HTML at module load. The `restore()` handle is invoked in
  // `finally` so multi-chunk workers (Cloud Run Jobs, Temporal activity
  // worker) don't leak chunk N's env into chunk N+1.
  const envRestore = applyRuntimeEnvSnapshot(encoder.runtimeEnv);

  try {
    // Synthesize a RenderJob the existing stages can consume. The chunk's
    // duration is its own frame count over fps — not the plan's full
    // duration — so the stages see this chunk as a self-contained render.
    const job = buildSyntheticRenderJob({
      fps: { num: plan.dimensions.fpsNum, den: plan.dimensions.fpsDen },
      quality: encoder.quality,
      format: plan.dimensions.format as "mp4" | "mov" | "png-sequence",
      crf: encoder.crf,
      bitrate: encoder.bitrate,
      hdrMode: "force-sdr",
      entryFile: "index.html",
    });
    job.totalFrames = framesInChunk;
    job.duration = (framesInChunk * plan.dimensions.fpsDen) / plan.dimensions.fpsNum;

    const cfg: EngineConfig = {
      ...resolveConfig(),
      browserGpuMode: "software",
      forceScreenshot: encoder.forceScreenshot,
    };

    // ── Per-chunk work + frames directories ──
    // Suffix workDir with pid + random bytes so concurrent invocations on
    // the SAME `(planDir, chunkIndex)` (e.g. a scheduler that double-fires
    // due to heartbeat skew) don't race on the same tmp tree. The output
    // path itself is still the caller's contract — concurrent writers to
    // `outputChunkPath` produce undefined bytes, but we don't make it worse
    // by also deleting their workDirs out from under them.
    const workDir = `${outputChunkPath}.work.${process.pid}.${randomBytes(4).toString("hex")}`;
    mkdirSync(workDir, { recursive: true });
    const framesDir = join(workDir, "captured-frames");
    mkdirSync(framesDir, { recursive: true });

    // ── File server with the seeded-random shim ──
    // `Math.random` / `crypto.getRandomValues` are seeded from virtual
    // time so retries are pixel-identical. Only distributed renders flip this.
    const fileServer: FileServerHandle = await createFileServer({
      projectDir: compiledDir,
      compiledDir,
      port: 0,
      preHeadScripts: [buildVirtualTimeShim({ seedRandomFromFrame: true })],
    });

    const captureOptions: CaptureOptions = {
      width: plan.dimensions.width,
      height: plan.dimensions.height,
      fps: { num: plan.dimensions.fpsNum, den: plan.dimensions.fpsDen },
      format: plan.dimensions.format === "mp4" ? "jpeg" : "png",
      quality: plan.dimensions.format === "mp4" ? 80 : undefined,
      deviceScaleFactor: encoder.deviceScaleFactor,
      // lock the BeginFrame warmup loop to a fixed iteration count so
      // `beginFrameTimeTicks` is host-independent. Only chunks ever set this.
      lockWarmupTicks: true,
    };

    // ── Browser + warmup ──
    let session: CaptureSession | null = null;
    let outputKind: "file" | "frame-dir";
    let framesEncoded = 0;
    try {
      session = await createCaptureSession(fileServer.url, framesDir, captureOptions, null, cfg);
      // SwiftShader assertion runs BEFORE initializeSession (which navigates to
      // the composition); on failure we tear down without ever touching the
      // composition URL. We pass `readWebGlVendorInfoFromCanvas` rather than
      // letting `assertSwiftShader` use its default `chrome://gpu` reader —
      // `chrome-headless-shell` serves chrome:// pages as empty documents,
      // which would trip a false-negative even when the GL backend is in fact
      // SwiftShader. The canvas + WEBGL_debug_renderer_info probe works on
      // any page (we navigate to about:blank inside the helper).
      await assertSwiftShader(session.page, readWebGlVendorInfoFromCanvas);
      await initializeSession(session);

      // Prime BeginFrame's `lastFrameCache` so the chunk's first real capture
      // reports `hasDamage` the same as an in-process render at the same
      // absolute frame would. Time is the chunk's first-frame absolute time.
      const startTime = (slice.startFrame * plan.dimensions.fpsDen) / plan.dimensions.fpsNum;
      await discardWarmupCapture(session, slice.startFrame, startTime);

      // ── Capture the chunk's range via runCaptureStage ──
      await runCaptureStage({
        fileServer,
        workDir,
        framesDir,
        job,
        totalFrames: framesInChunk,
        cfg,
        forceScreenshot: encoder.forceScreenshot,
        log,
        workerCount: 1,
        // Pass the pre-warmed session through as `probeSession` so captureStage
        // reuses it via `prepareCaptureSessionForReuse` instead of spinning up
        // a fresh browser. The stage closes the session in its `finally`,
        // so we MUST clear our own reference here to avoid a double-close.
        probeSession: session,
        needsAlpha: plan.dimensions.format !== "mp4",
        captureAttempts: [],
        buildCaptureOptions: () => captureOptions,
        createRenderVideoFrameInjector: () => null,
        abortSignal: undefined,
        assertNotAborted: () => {},
        frameRange: { startFrame: slice.startFrame, endFrame: slice.endFrame },
      });
      // captureStage closes the session it consumed.
      session = null;
      framesEncoded = framesInChunk;

      // ── Encode the chunk ──
      const isPngSequence = plan.dimensions.format === "png-sequence";
      outputKind = isPngSequence ? "frame-dir" : "file";
      // For mp4/mov we use the standard preset machinery; the locked encoder
      // values come from `meta/encoder.json` and the `lockGopForChunkConcat`
      // toggle is the only Phase-2 flag that flips on at this site.
      // png-sequence has no encoder, but `runEncodeStage` still reads
      // `preset.quality` for bookkeeping (it never reaches ffmpeg on the
      // pngseq branch). Fall back to the mp4 preset shape — same trick
      // `renderOrchestrator` plays.
      const presetFormat: "mp4" | "mov" | "webm" = isPngSequence
        ? "mp4"
        : (plan.dimensions.format as "mp4" | "mov");
      const preset = getEncoderPreset(job.config.quality, presetFormat, undefined);
      const effectiveQuality = encoder.crf ?? preset.quality;
      const effectiveBitrate = encoder.crf != null ? undefined : encoder.bitrate;
      // For non-pngseq, encodeStage writes to `outputPath` when `isPngSequence`
      // is false. `videoOnlyPath` is the encoder's direct output (no mux —
      // mux happens in assemble()).
      const videoOnlyPath = outputChunkPath;
      if (isPngSequence) {
        if (!existsSync(outputChunkPath)) mkdirSync(outputChunkPath, { recursive: true });
      } else {
        const outDir = join(outputChunkPath, "..");
        if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });
      }

      await runEncodeStage({
        job,
        log,
        outputPath: outputChunkPath,
        framesDir,
        videoOnlyPath,
        width: plan.dimensions.width * encoder.deviceScaleFactor,
        height: plan.dimensions.height * encoder.deviceScaleFactor,
        needsAlpha: plan.dimensions.format !== "mp4",
        // Each chunk produces video only — audio is muxed once at assemble
        // time. Suppressing `hasAudio` skips the png-sequence audio sidecar
        // AND the mp4 audio mux.
        hasAudio: false,
        isPngSequence,
        preset,
        effectiveQuality,
        effectiveBitrate,
        // Distributed chunks emit a single ffmpeg call per chunk; the
        // in-process per-chunk-within-chunk path would re-split our
        // already-chunked work.
        enableChunkedEncode: false,
        chunkedEncodeSize: framesInChunk,
        abortSignal: undefined,
        assertNotAborted: () => {},
        // GOP === framesInChunk + force-keyframe at frame 0 → the chunk's
        // first frame is an IDR keyframe and concat-copy at assemble time
        // round-trips losslessly.
        lockGopForChunkConcat: !isPngSequence,
        gopSize: framesInChunk,
      });
    } finally {
      // Cleanest path: captureStage closed the session for us. The defensive
      // close handles error paths where we threw before delegating.
      if (session) {
        try {
          await closeCaptureSession(session);
        } catch (err) {
          log.warn("[renderChunk] error closing capture session in finally", {
            error: err instanceof Error ? err.message : String(err),
          });
        }
      }
      fileServer.close();
      // Leave the temp work dir on failure (helps debugging); remove it on
      // success below.
    }

    // ── Hash the output + write the perf sidecar ──
    const sha256 = hashChunkOutput(outputChunkPath, outputKind);
    const durationMs = Date.now() - start;
    const perfPath = `${outputChunkPath}.perf.json`;
    const perfPayload = {
      planHash: plan.planHash,
      chunkIndex,
      startFrame: slice.startFrame,
      endFrame: slice.endFrame,
      framesEncoded,
      durationMs,
      sha256,
      outputKind,
      producerVersion: plan.producerVersion,
      ffmpegVersion,
    };
    writeFileSync(perfPath, `${JSON.stringify(perfPayload, null, 2)}\n`, "utf-8");

    // Clean up only after the hash + perf sidecar landed. Any failure above
    // leaves the framesDir in place for inspection.
    try {
      rmSync(workDir, { recursive: true, force: true });
    } catch (err) {
      log.warn("[renderChunk] failed to remove work dir", {
        workDir,
        error: err instanceof Error ? err.message : String(err),
      });
    }

    return {
      outputPath: outputChunkPath,
      outputKind,
      framesEncoded,
      sha256,
      durationMs,
      perfPath,
    };
  } finally {
    // Restore the controller's runtime env even on the error path so the
    // next chunk on the same process boots from a clean env.
    envRestore.restore();
  }
}
