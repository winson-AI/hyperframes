/**
 * Tests for plan-time validators. Each validator pins both branches:
 *
 *   - PASS — the config is acceptable; no throw.
 *   - FAIL — the config trips a banned-in-distributed-mode rule; throws
 *     PlanValidationError with the expected typed `code`.
 */

import { describe, expect, it } from "bun:test";
import {
  BROWSER_GPU_NOT_SOFTWARE,
  PlanValidationError,
  validateNoGpuEncode,
} from "./planValidation.js";

describe("PlanValidationError", () => {
  it("preserves the typed `code` field", () => {
    const err = new PlanValidationError("EXAMPLE_CODE", "msg");
    expect(err.code).toBe("EXAMPLE_CODE");
    expect(err.message).toBe("msg");
    expect(err.name).toBe("PlanValidationError");
    expect(err).toBeInstanceOf(Error);
  });
});

describe("validateNoGpuEncode", () => {
  it("accepts a software-only config (no fields set)", () => {
    expect(() => validateNoGpuEncode({})).not.toThrow();
  });

  it("accepts useGpu=false + browserGpuMode='software'", () => {
    expect(() => validateNoGpuEncode({ useGpu: false, browserGpuMode: "software" })).not.toThrow();
  });

  it("accepts useGpu=undefined (in-process default) + browserGpuMode='software'", () => {
    expect(() => validateNoGpuEncode({ browserGpuMode: "software" })).not.toThrow();
  });

  it("throws BROWSER_GPU_NOT_SOFTWARE when useGpu === true", () => {
    let caught: unknown;
    try {
      validateNoGpuEncode({ useGpu: true });
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(PlanValidationError);
    expect((caught as PlanValidationError).code).toBe(BROWSER_GPU_NOT_SOFTWARE);
    expect((caught as PlanValidationError).code).toBe("BROWSER_GPU_NOT_SOFTWARE");
    expect((caught as Error).message).toContain("GPU encode is banned");
    expect((caught as Error).message).toContain("useGpu === true");
  });

  it("throws BROWSER_GPU_NOT_SOFTWARE when browserGpuMode === 'auto'", () => {
    let caught: unknown;
    try {
      validateNoGpuEncode({ browserGpuMode: "auto" });
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(PlanValidationError);
    expect((caught as PlanValidationError).code).toBe(BROWSER_GPU_NOT_SOFTWARE);
    expect((caught as Error).message).toContain("Hardware browser GPU is banned");
    expect((caught as Error).message).toContain(`"auto"`);
  });

  it("throws BROWSER_GPU_NOT_SOFTWARE for any non-'software' browserGpuMode value", () => {
    for (const mode of ["hardware", "discrete", "any", "swiftshader-fallback"]) {
      let caught: unknown;
      try {
        validateNoGpuEncode({ browserGpuMode: mode });
      } catch (err) {
        caught = err;
      }
      expect(caught).toBeInstanceOf(PlanValidationError);
      expect((caught as PlanValidationError).code).toBe(BROWSER_GPU_NOT_SOFTWARE);
    }
  });

  it("checks useGpu BEFORE browserGpuMode so the useGpu message wins when both trip", () => {
    let caught: unknown;
    try {
      validateNoGpuEncode({ useGpu: true, browserGpuMode: "auto" });
    } catch (err) {
      caught = err;
    }
    expect(caught).toBeInstanceOf(PlanValidationError);
    expect((caught as Error).message).toContain("GPU encode is banned");
  });
});
