/**
 * Plan-time validators for the distributed render pipeline. Each validator
 * is invoked before freezing the plan, so banned configurations fail fast
 * with a typed non-retryable error instead of being baked into a planDir
 * and only surfacing on the chunk worker.
 */

import { BROWSER_GPU_NOT_SOFTWARE } from "@hyperframes/engine";

/**
 * Re-export the BROWSER_GPU_NOT_SOFTWARE code so distributed adapters and
 * Step Functions / Temporal retry policies can match it without a
 * cross-package import.
 */
export { BROWSER_GPU_NOT_SOFTWARE } from "@hyperframes/engine";

/**
 * Typed plan-validation error. Workflow adapters key retry policies off the
 * `code` field to mark errors as non-retryable.
 */
export class PlanValidationError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "PlanValidationError";
    this.code = code;
  }
}

/**
 * Subset of the merged plan / engine / render config that the GPU validator
 * inspects. Both `useGpu` (RenderConfig) and `browserGpuMode` (EngineConfig)
 * are optional so callers can pass any of the surrounding config shapes
 * without an adapter layer.
 *
 *   - `useGpu === true` → encoder GPU acceleration (NVENC/QSV/VAAPI). Banned
 *     because GPU encoders produce non-byte-identical output across machines.
 *   - `browserGpuMode !== "software"` → headless Chrome's WebGL is allowed
 *     to use hardware GL. Banned because hardware GL is bitwise unstable
 *     across drivers. Pairs with the runtime `assertSwiftShader` check that
 *     catches workers whose environment ignores Chrome's `--use-gl=swiftshader`.
 */
export interface ValidateNoGpuEncodeInput {
  useGpu?: boolean;
  browserGpuMode?: string;
}

/**
 * Reject any config that would let GPU encode or hardware-GL slip into a
 * distributed render. Throws {@link PlanValidationError} with
 * `code === BROWSER_GPU_NOT_SOFTWARE` when either gate trips. The message
 * names the offending field so the caller can surface a clean error.
 */
export function validateNoGpuEncode(config: ValidateNoGpuEncodeInput): void {
  if (config.useGpu === true) {
    throw new PlanValidationError(
      BROWSER_GPU_NOT_SOFTWARE,
      "[planValidation] GPU encode is banned in distributed mode: " +
        "config.useGpu === true. " +
        "Distributed retries must be byte-identical, but NVENC/QSV/VAAPI " +
        "produce different output across machines. Set useGpu=false (the " +
        "default) — software libx264/libx265 is the only supported encoder " +
        "in distributed mode.",
    );
  }
  if (config.browserGpuMode !== undefined && config.browserGpuMode !== "software") {
    throw new PlanValidationError(
      BROWSER_GPU_NOT_SOFTWARE,
      `[planValidation] Hardware browser GPU is banned in distributed mode: ` +
        `config.browserGpuMode === ${JSON.stringify(config.browserGpuMode)}. ` +
        `Hardware GL is bitwise unstable across drivers. Set browserGpuMode="software" ` +
        `so Chrome launches with --use-gl=swiftshader.`,
    );
  }
}
