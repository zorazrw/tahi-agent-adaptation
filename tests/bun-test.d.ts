/** Minimal shim so the IDE resolves ``bun:test`` (runtime is provided by Bun). */
declare module "bun:test" {
  type TestFn = () => void | Promise<void>;

  export function describe(name: string, fn: TestFn): void;
  export function test(name: string, fn: TestFn): void;
  export function afterEach(fn: TestFn): void;

  export const expect: <T>(actual: T) => {
    toBe(expected: T): void;
    toEqual(expected: unknown): void;
    toContain(expected: string): void;
    toHaveLength(expected: number): void;
  };
}
