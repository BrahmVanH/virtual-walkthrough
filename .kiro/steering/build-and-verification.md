---
inclusion: always
---

# Build & verification: the user owns the loop

## Rule 1 (highest priority): never build, check, lint, or test

Never run any of these, in the foreground, as a background process, or as one
step of a chained command. This overrides any instinct to verify work by
executing it. The user builds asynchronously in their own terminal and reports
results back.

JS / SvelteKit side:

- `npm run dev`, `npm run build`, `npm run preview`, `vite`, `vite build`
- `npm run check`, `npm run check:watch`, `svelte-kit sync`, `svelte-check`, `tsc`
- `npm run lint`, `eslint`
- `npm test`, `vitest`, `playwright`, any test runner
- `npm install` / `npm ci` (also triggers `prepare` → `svelte-kit sync`)

Rust / WASM side (`pano-stitcher/`):

- `cargo build|check|clippy|test|bench|run|fix|doc`, `rustc`
- `wasm-pack build`, `npm run wasm`, `npm run wasm:netlify`,
  `npm run modulize-wasm`, `bash build-netlify.sh`
- any script or wrapper that shells out to the above

Do not suggest running these "for" the user either. Name the command and let
them run it.

## Rule 2: never manage local services

Dev servers, `netlify dev`, containers (`docker`/`podman` start, stop, run,
compose, prune) belong to the user. Read-only inspection (`docker ps -a`,
`docker logs`) is fine. If a service is down, report what the read-only
commands show and stop. Local data is the user's data: never drop, recreate,
or bulk-delete anything to get a clean slate.

## Rule 3: verify statically instead

1. Make the change.
2. Re-read the edited files against surrounding code: imports resolve, types
   line up, trait bounds hold, match arms are exhaustive, public signatures
   still match call sites (including the JS side of any `#[wasm_bindgen]`
   export), tests reference things that exist.
3. Use language-server diagnostics on touched files where available
   (rust-analyzer, svelte/TS language server). That is the one allowed
   compile-adjacent check because it does not spawn a build.
4. For `.svelte` files, run the Svelte MCP `svelte-autofixer` (see `AGENTS.md`).

Read-only shell commands unrelated to building (`ls`, `cat`, `git status`,
`git diff`) are fine when a dedicated tool can't do the job.

## Rule 4: hand off explicitly

End the turn with the narrowest command that covers the change and what to
watch for. Examples:

> Ready for you to verify. Run `cargo test --manifest-path pano-stitcher/Cargo.toml`.
> I changed `estimate_homography` to return `Result`, so watch the call site in `lib.rs`.

> Changes are in. Run `npm run wasm` then `npm run check`; the new export
> `stitch_row` should appear in `static/wasm/pano_stitcher.d.ts`.

## Rule 5: be honest about verification state

Never call unbuilt code working, passing, compiling, or verified. Say what I
checked and what I did not: "Change is in, not yet compiled." When the user
reports errors, fix them and hand back for another run.
