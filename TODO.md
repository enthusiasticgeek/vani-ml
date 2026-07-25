# vani-ml — TODO

> Full scope, architecture decisions, and per-phase risk notes are in
> [kosh-index/ROADMAP.md](https://github.com/enthusiasticgeek/kosh-index/blob/main/ROADMAP.md#planned-ml-tier-scoped-2026-07-25)
> ("`vani-ml` scoping breakdown"). This file is just the actionable checklist for
> this repo. Optional tier -- confirm scope before starting each phase.

---

## v0.1.0 — Classical ML ✅ implemented 2026-07-25

Depends on: vani-probability (^0.4.2, published), vani-optimize (^0.1.4, published).
Vendored via `vanic vendor`.

- [x] Linear regression — `linreg_add_intercept`/`linreg_fit`/`linreg_predict`/
      `linreg_r_squared`, thin wrappers over `probability::mlr_*`. Test:
      exact-fit recovery on a noise-free synthetic dataset.
- [x] Logistic regression — `logreg_fit`/`logreg_predict_proba`/`logreg_predict`,
      cross-entropy loss. **Not** built on vani-optimize's
      `gradient_descent_fixed`/`backtracking` — their fixed
      `fn(ref Vec<f64>, i64) -> f64` objective/gradient signature can't carry
      the training data through without a ref-capturing closure (unsupported,
      compiler path-D). Own small dedicated gradient-descent loop instead,
      same algorithm shape. Test: converges to perfect separation on a
      linearly-separable 1D dataset.
- [x] k-means clustering — `kmeans_fit`/`kmeans_predict`/`kmeans_inertia` +
      `KMeansResult`, Lloyd's algorithm, random-point init, seeded via
      `seed_rng`/`rand_in_range`. Genuinely new code, no existing kosh
      package covers this. Test: exact label/centroid/inertia recovery on
      two well-separated 2D clusters.
- [x] Train/test split — `train_test_split` + `TrainTestSplit`, seeded
      Fisher-Yates shuffle. Test: split-size and partition-sum invariants.
- [x] Core metrics — `mse`/`accuracy`/`precision`/`recall`/`f1_score`. Test:
      hand-computed confusion matrix, exact expected values.
- [x] Tests: 4 files under `tests/`, all pass `vanic test` (LLVM backend),
      full `vanic check` SMT verification, and `vanic audit-safety`
      (`#[bounded_stack]` coverage complete on all 20 src functions).
- [x] Example: `examples/ml_demo.vani`, verified on both LLVM and C backends.

**Found along the way**: a real `vanic` LLVM-backend crash on a standalone
unary-minus float literal (e.g. `-3.0` as a `vec()` argument) — filed
upstream as BUG-6 in `vani-compiler/docs/TODO_CURRENT.md`, not fixed.
Worked around with `0.0 - 3.0` throughout, per the existing "no unary
minus" convention.

**Not yet done**: `vanic publish` — stopping here since publishing is a
shared/irreversible action; get an explicit go-ahead before publishing.

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
