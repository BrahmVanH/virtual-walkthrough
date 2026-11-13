---
inclusion: always
---

# Testing policy

## Rule: tests accompany all feature work

- **New features** include tests for the happy path and key edge cases.
- **Changed features** update related tests to match, plus new cases for any new
  branches or failure modes.
- **Bug fixes** include a regression test that fails without the fix.
- Write the tests, never run them (see `build-and-verification.md`). Hand off
  the narrowest command that runs them.

## Rust (`pano-stitcher/`)

- Unit tests go in a `#[cfg(test)] mod tests` at the bottom of the module they
  cover (`homography.rs`, `lib.rs`, and so on).
- Keep the stitching math testable as plain Rust: pure functions over
  `nalgebra` types and byte slices, with the `#[wasm_bindgen]` layer kept thin so
  native `cargo test` covers the logic without a browser.
- Prefer synthetic, deterministic inputs: known homographies applied to
  generated point sets, small generated images. RANSAC is seeded (`Pcg64`
  with a fixed seed), so assert on exact outcomes where that holds.
- Compare floats with a tolerance, never `==`.
- Test fixtures (real photos) go in `pano-stitcher/tests/fixtures/` only when
  synthetic data can't exercise the case; keep them small.
- Handoff command: `cargo test --manifest-path pano-stitcher/Cargo.toml`
  (add a test-name filter when the change is narrow).

### WASM boundary tests

- Anything that only exists inside a JS host (`JsValue`, `JsError`,
  serde-wasm-bindgen output shape such as `null` vs `undefined` and camelCase
  keys) can't run under native `cargo test`. Test it with `wasm-bindgen-test`
  (wasm32-only dev-dependency) in a `#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests` after `mod tests`, using `#[wasm_bindgen_test]`.
- Keep these few and focused on the boundary; push logic into native-testable
  functions and call them from the export.
- `wasm-bindgen-test` pins an exact `wasm-bindgen` version. When `Cargo.lock`
  bumps `wasm-bindgen`, bump `wasm-bindgen-test` to the matching release.
- Handoff command: `wasm-pack test --node pano-stitcher` (runs in Node, no
  browser needed).

## SvelteKit (`src/`)

- No JS test runner is installed yet. Before writing the first JS test, ask the
  user to confirm adding Vitest (and whether they want `@testing-library/svelte`
  or browser-mode tests). Don't add the dependency unasked.
- Once set up: tests sit next to the module as `<filename>.test.ts`
  (e.g. `wasm-loader.test.ts` next to `wasm-loader.ts`), use `describe`/`it`
  with behavior-focused names, and mock only what is necessary (the WASM module,
  `$app/environment`, browser APIs).
- No shared test helpers or `$env` stubs exist yet. When the first one is
  needed, put helpers in `src/test/helpers/` and env stubs in `src/test/stubs/`,
  then update this file to list them.
