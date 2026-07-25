# vani-ml — TODO

> Full scope, architecture decisions, and per-phase risk notes are in
> [kosh-index/ROADMAP.md](https://github.com/enthusiasticgeek/kosh-index/blob/main/ROADMAP.md#planned-ml-tier-scoped-2026-07-25)
> ("`vani-ml` scoping breakdown"). This file is just the actionable checklist for
> this repo. Optional tier -- confirm scope before starting each phase.

---

## v0.1.0 — Classical ML (not started)

Depends on: vani-probability (^0.4.2, published), vani-optimize (^0.1.4, published).
Not yet vendored — run `vanic vendor` before first build.

- [ ] Linear regression — thin wrapper over vani-probability's existing MLR
- [ ] Logistic regression — cross-entropy loss + vani-optimize's gradient descent
      (fixed-step or Armijo-backtracking)
- [ ] k-means clustering — new implementation, no existing package covers this
- [ ] Train/test split
- [ ] Core metrics: accuracy, MSE, precision/recall
- [ ] Tests: cross-check regression/logistic-regression against known reference
      values (e.g. sklearn or textbook cases), same discipline as the rest of
      this ecosystem — not just hand-picked examples
- [ ] Example(s) in `examples/`

## v0.2.0 — Data utilities (not started)

Depends on: v0.1.0.

- [ ] Feature scaling: standardize, normalize
- [ ] One-hot encoding
- [ ] `Dataset` struct — row-major `Vec<f64>` features + labels, matching
      vani-matrix/vani-tensor's row-major convention
- [ ] k-fold cross-validation

## v0.3.0 — Autodiff core (not started) — highest-risk phase

Depends on: v0.2.0, vani-tensor (value storage; not yet a declared dep — add when
this phase starts).

- [ ] Flat arena: `Vec<Node>` + `i64` child indices + node-kind tags (same pattern
      as vani-symbolic's expression arena)
- [ ] Forward evaluation over the arena
- [ ] Reverse-mode backward pass — gradients accumulated into a `mut ref
      Vec<f64>` buffer passed explicitly through every call, **not** a
      ref-capturing closure (unsupported — compiler path-D, deferred
      indefinitely; see this repo's README and kosh-index/ROADMAP.md)
- [ ] Validation: finite-difference gradient checking against every node kind
      (numeric gradient ≈ autodiff gradient within tolerance) — this is the
      primary correctness gate for this phase, not hand-picked examples

## v0.4.0 — Layers, activations, losses (not started)

Depends on: v0.3.0.

- [ ] Dense/linear layer (matmul + bias via vani-matrix/vani-tensor)
- [ ] Activations as graph node kinds: relu, sigmoid, tanh, softmax
- [ ] Losses as graph node kinds: MSE, cross-entropy
- [ ] Each new node kind validated with the same finite-difference check as v0.3.0

## v0.5.0 — Optimizers over the graph (not started)

Depends on: v0.3.0.

- [ ] SGD
- [ ] Momentum
- [ ] Adam
- [ ] Note: vani-optimize's existing solvers don't fit this signature
      (per-graph gradient vector, not `fn(ref Vec<f64>) -> f64`) — new code,
      same underlying math, not a straight reuse

## v0.6.0+ — Training-loop utilities (optional/stretch, not started)

Depends on: v0.1.0-v0.5.0.

- [ ] Training-loop helper, batching
- [ ] A couple of worked small-MLP examples end-to-end
- [ ] Open-ended — revisit scope once the core phases are shipped and actually used

---

## Future / explicitly out of scope for now

- CNN/RNN-specific layers — not scoped; revisit only if a real use case shows up
- GPU/SIMD-accelerated ops — depends on vāṇी compiler SIMD support maturity,
  tracked separately in vani-compiler's docs
