# Configuration architecture

IDCS configuration is moving from runtime peer-to-peer synchronization to an
immutable, locally loaded deployment bundle.

## Authority and deployment

Version-controlled YAML is the authoring source. A deployment procedure places
the intended commit and any explicitly selected site override on each host.
Each process reads its ordered files once at startup. Runtime processes never
select a winner by modification time, rewrite source YAML, or create sync state
beside source files.

Layer order is explicit: the base file is first and later site/runtime layers
override it. Mappings merge recursively; sequences and scalar values replace
the earlier value. This prevents a small nested override from silently deleting
unrelated settings in the same section.

`common.config.load_config_bundle()` is the migration boundary. It provides:

- strict UTF-8 and YAML parsing with duplicate-key rejection;
- string-keyed, JSON-compatible value validation;
- required-section validation at each process boundary;
- a recursively immutable resolved mapping;
- SHA-256 for each exact source file and for canonical resolved content;
- no filesystem writes.

The resolved digest identifies behaviorally relevant configuration. Exact
source hashes preserve provenance even when comments or layer organization
produce the same resolved values.

## Runtime contract

Every long-running process should expose `config_digest` and
`config_sources` containing absolute path, byte size, and exact-file SHA-256.
Readiness means configuration validation and the process-specific readiness
gate both passed. A digest mismatch between hosts is observable state, not a
trigger for either peer to mutate the other.

## Migration sequence

1. Passive DeepStream runtime (no actuator authority).
2. PC stream and UI processes.
3. Encoder and manual-input services.
4. Controller and serial-owner services.
5. Remove config-sync startup protocols, marker/lock code, and endpoint config.

During migration, only an explicitly migrated process uses the immutable
loader. Legacy processes retain their current behavior so a partial rollout
does not silently change startup semantics.

## Verification

Loader tests use temporary local files and assert that no sidecar files appear.
Runtime `--check` is the first integration gate: it resolves all layers,
validates required process sections and file references, prints provenance, and
opens no sockets, video devices, or control interfaces.

Deployment acceptance must record the Git commit, clean/dirty status, resolved
config digest, source hashes, host identity, and process-specific artifacts as
specified in `docs/verification_strategy.md`.
