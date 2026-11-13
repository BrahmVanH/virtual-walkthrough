---
inclusion: always
---

# Comment sparingly

Applies to Rust, TypeScript, Svelte, and CSS alike. Default to no comment.
Write the code so the comment is unnecessary before writing the comment.

## Do not comment

- Restatements of the code (`// increment counter` above `counter += 1`)
- Names already carried by the identifier, type, or signature
- Section banners and decorative dividers
- Change logs, dates, ticket numbers, or "was X, now Y" notes; that is git's job
- Notes addressed to the user ("your loop re-extracted every image"); put that
  in the chat reply, not the source
- Commented-out code. Delete it.
- Doc comments on private items whose body is a few obvious lines

## Do comment

Only when a reader who knows the language still can't recover it from the code:

- Why a non-obvious approach was chosen, especially when the obvious one fails
- Workarounds for upstream bugs or platform quirks (wasm32, browser, Netlify),
  with a link or issue ref
- Invariants the compiler can't express, including `// SAFETY:` on `unsafe`
- Units, ranges, encodings, coordinate spaces, and ordering guarantees the type
  doesn't state (pixels vs normalized coords, degrees vs radians, RGBA order)
- Subtle algorithmic reasoning: why a threshold has its value, why an `unwrap`
  can't panic, why a numerical step is needed
- Public API docs (`///` on exported Rust items, especially `#[wasm_bindgen]`
  exports; TSDoc on exported `$lib` functions), kept to intent and contract

## Style

Terse, one line where one line does. Explain intent, not mechanics. Place the
comment directly above the code it covers, not trailing a long line.

```rust
// Hartley normalization; the DLT is numerically unstable on raw pixel coords.
```

## Editing existing code

Match the surrounding file when it is consistently documented. Don't add
commentary to code I'm only passing through, and don't strip existing comments
unless my change makes them wrong. When a change makes a comment stale, update
or delete it in the same edit.
