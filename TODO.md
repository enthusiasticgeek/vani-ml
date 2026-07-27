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

**Published** 2026-07-26 (this note was stale until 2026-07-27 -- v0.1.0
was already live in kosh-index by the time v0.2.0's work below started).

## v0.2.0 — Data utilities ✅ implemented 2026-07-27

Depends on: v0.1.0.

- [x] Feature scaling: `StandardScaler`/`standardize_fit`/
      `standardize_apply` (z-score, population std) and
      `MinMaxScaler`/`normalize_fit`/`normalize_apply` (min-max to
      [0,1]). Deliberately a fit/apply split, not a single all-in-one
      transform, so a caller can fit on training data and apply the SAME
      parameters to held-out test data (fitting separately on train and
      test would leak test-set statistics into the transform). Both
      guard the constant-column case (std/range ~0) rather than dividing
      by ~0 -- standardize centers without scaling, normalize maps to
      0.0.
- [x] `one_hot_encode(labels, n_classes)` -- row-major
      `len(labels) x n_classes` Vec<f64>.
- [x] `Dataset` struct — row-major `Vec<f64>` features + labels, matching
      vani-matrix/vani-tensor's row-major convention. A thin bundle (not
      a new abstraction) so functions that compose multiple datasets
      (like `k_fold_split`) can take/return one value instead of four.
- [x] `k_fold_split(X_flat, y, n_obs, n_dim, k, fold, seed)` -- k-fold
      cross-validation SPLITTING (mirroring `train_test_split`'s own
      scope), not a full train/eval harness: this package has no generic
      "trainable model" abstraction (linreg/logreg/kmeans each have
      distinct fit signatures), so a caller loops `fold` from 0 to k-1
      themselves and calls whichever model's fit/predict they want per
      iteration, same "caller loops themselves, no closures needed"
      convention as `vani-vectorcalc`'s line integrals. Same seeded
      Fisher-Yates shuffle as `train_test_split`; calling this k times
      with the same seed reproduces the identical underlying shuffle
      each time (since `seed_rng` fully resets the RNG state), so the k
      folds are a genuine partition, not k independent random subsets.
      Remainder from `n_obs` not dividing evenly by `k` is distributed
      one-per-fold across the first `n_obs % k` folds.
- [x] `tests/test_data_utils.vani` -- scaling verified against
      hand-computed stats (mean/std/min/max) including the constant-
      column guard path for both scalers; one-hot encoding on a 4-label,
      3-class example; k-fold split (n_obs=10, k=3, fold sizes 4/3/3)
      verified via the same partition-invariant composed check
      `train_test_split`'s own test uses (every original value appears
      exactly once across all k folds' test partitions, and every
      individual fold's train+test also sums back to the whole set).
      Full suite + `vanic audit-safety` re-verified on both backends.

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
