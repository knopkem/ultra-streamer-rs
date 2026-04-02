## Code Quality

- Write **production-ready Rust** — no placeholder logic, no `todo!()` left in non-test code, no `unwrap()` or `expect()` outside of tests or `main` startup validation where a panic is acceptable.
- Prefer `?` for error propagation. Define domain-specific error types with `thiserror`. Never use `anyhow` in library crates; `anyhow` is acceptable only in binary entry points.
- All public items must have doc comments (`///`). Include at least one `# Example` block for non-trivial public APIs.
- No `clippy` warnings — code must pass `cargo clippy -- -D warnings` clean. Apply `#[allow(...)]` only when genuinely necessary and always with a comment explaining why.
- Format all code with `rustfmt` (default settings). Never submit unformatted code.
- Avoid `unsafe` unless interfacing with C FFI (e.g., OpenJPEG). Every `unsafe` block must have a `// SAFETY:` comment explaining the invariants upheld.

---

## Rust Patterns

Apply idiomatic Rust patterns consistently:

- **Newtype pattern** for domain identifiers — prevents mixing up UIDs at the type level.
- **Builder pattern** for structs with many optional fields (e.g., query builders, config structs). Implement via a dedicated `XxxBuilder` struct with a consuming `build() -> Result<Xxx>`.
- **Typestate pattern** for protocol state machines (e.g., DIMSE association lifecycle: `Association<Unassociated>` → `Association<Established>`).
- **`From`/`Into`/`TryFrom`/`TryInto`** for all conversions between domain types and external types.
- **`Display` + `Error`** implementations on all error types.
- **`Default`** on config and option structs where zero-value defaults are meaningful.
- Prefer **`Arc<dyn Trait>`** for shared, injectable dependencies (`MetadataStore`, `BlobStore`) — enables testing with mocks.
- Use **`tokio::sync`** primitives (`RwLock`, `Mutex`, `broadcast`, `mpsc`) over `std::sync` in async code.
- Leverage **`tower` middleware** (tracing, timeout, rate-limit) for Axum routes rather than duplicating cross-cutting logic in handlers.
- Prefer **`bytes::Bytes`** for zero-copy binary data passing between components (DICOM pixel data, multipart bodies).

---

## Testing (Mandatory)

Testing is not optional. Every PR must include tests. The bar:

### Unit Tests
- Every module with non-trivial logic must have a `#[cfg(test)] mod tests { ... }` block in the same file.
- Test the happy path, error paths, and edge cases.
- Use **`rstest`** for parameterised tests and fixtures.
- Mock trait dependencies with **`mockall`** — never reach out to real databases or network in unit tests.

### Integration Tests
- Place integration tests in `tests/` at the crate root.

### End-to-End Tests
- `tests/integration/` at workspace root contains E2E tests

### Test Hygiene
- No `#[ignore]` without a tracking issue reference in the comment.
- All tests must be deterministic — no reliance on system time, random values, or port availability without explicit seeding/allocation.
- Use **`tokio::test`** for async tests. Use **`#[tokio::test(flavor = "multi_thread")]`** when testing concurrency behaviour.
- Assert error types precisely — match the variant, not just `is_err()`.

---

## Error Handling

- Library crates define their own `Error` enum with `thiserror`.
- Errors must be meaningful to the caller — wrap lower-level errors with context using `#[from]` or `.map_err(|e| Error::StoreFailed { source: e, uid: uid.to_string() })`.
- Never silently swallow errors. Log with `tracing::error!` at the boundary where you decide not to propagate.

---

## Async & Concurrency

- All I/O is async. Blocking operations (file I/O, CPU-heavy codec work) must be dispatched via `tokio::task::spawn_blocking`.
- Avoid holding locks across `.await` points. Prefer scoped lock guards or restructure code to release before awaiting.
- Cancellation safety: document any `async fn` that is NOT cancellation-safe with a `# Cancellation Safety` section in its doc comment.

---

## Logging & Tracing

- Use `tracing` spans and events throughout, not `println!` or `log!`.
- Instrument all service-layer functions with `#[tracing::instrument(skip(self), err)]`.

