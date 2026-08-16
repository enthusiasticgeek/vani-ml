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

## v0.3.0 — Autodiff core ✅ implemented 2026-08-16 — highest-risk phase

Depends on: v0.2.0. **vani-tensor deliberately NOT added as a dependency**
(see below) — the roadmap's "value storage" note turned out not to apply
to this phase's actual design.

- [x] Flat arena: `Vec<GraphNode>` + `i64` child indices + node-kind tags
      (`GraphNode { kind, value, left, right }`, same pattern as
      vani-symbolic's `ExprNode`). **Departure from vani-symbolic**: no
      symbol table — autodiff graphs are built directly via function
      calls (never parsed from names), so a caller who wants to reuse a
      value just reuses the `i64` index a constructor already returned
      them. This means the SAME index can legitimately be two different
      nodes' child — a real DAG, not just a tree — which reverse-mode
      autodiff must handle correctly (gradient contributions from every
      use of a value must SUM, not overwrite). Kind tags: `0=Const
      1=Param 2=Add 3=Sub 4=Mul 5=Div 6=Neg`. `Pow` deliberately left out
      of scope — `f(x)=x*x` (same index as both `graph_mul` operands)
      already exercises the DAG-accumulation case without it.
- [x] Forward evaluation over the arena — `graph_forward`, a single
      ascending `while` loop (no recursion): because the arena is
      append-only, a node's children always have a lower index than the
      node itself, so evaluating index order 0..n-1 is always correct.
- [x] Reverse-mode backward pass — `graph_backward`, a single
      **descending** `while` loop from n-1 to 0, seeded with
      `grads[root]=1.0`: by the time node `i` is processed, every
      possible consumer of node `i` (strictly higher index) has already
      run and added its contribution via `+=`, so `grads[i]` is
      guaranteed final. Standard reverse-mode-AD algorithm (same one
      micrograd/PyTorch's tape-based autograd use); the append-only
      arena gets the correct traversal order for free, no topo-sort
      needed. Gradients accumulated into a freshly-returned `Vec<f64>`
      (not a `mut ref` out-parameter — this ecosystem's dominant
      "return a new Vec" convention, e.g. `tensor_zeros`/`tensor_add`);
      the roadmap's "`mut ref Vec<f64>` buffer threaded explicitly"
      language is about avoiding ref-capturing closures internally, not
      the public API's shape. No ref-capturing closures anywhere in this
      phase (compiler path-D, deferred indefinitely).
- [x] Validation: finite-difference gradient checking against every node
      kind (`tests/test_graph_core.vani`, 9 tests), genuinely
      cross-checked against `calculus::diff_central` (same cross-package
      validation vani-symbolic's own v0.3.0 used against the same
      function) — one test per kind (Add/Sub/Mul/Div/Neg), a dedicated
      **DAG/shared-node accumulation** test (`f(x)=x*x`, catches the
      exact "overwrite instead of accumulate" bug class this design has
      to get right), a **composed** test mixing all three binary ops in
      one graph (`f(x,y)=(x*y+x)/y`, checked at both params), a Const-leaf
      test, and a `graph_set_value` test. All 9 pass on both backends.
      `examples/autodiff_demo.vani` (new file, both backends verified)
      walks through the same composed example plus the shared-node case.

**Found during implementation, not before scoping**: the checker doesn't
reborrow a `mut ref` binding as `ref` at a call site — same limitation
vani-symbolic's own `sym_add` already documents hitting (confirmed with
the identical error message). This ruled out a shared finite-difference
helper taking `arena: mut ref Vec<GraphNode>` (would need to call both
`graph_set_value`, which wants `mut ref`, and `graph_forward`, which
wants `ref`, on the same re-borrowed parameter) — the perturb/forward/
restore logic is inlined into every test instead, matching
vani-symbolic's own documented workaround exactly rather than
rediscovering a new one.

**Scope note on `vani-tensor`**: the roadmap listed vani-tensor ("value
storage") as a v0.3.0 dependency, but this design stores one `f64` per
node directly in the arena/tape — it never touches vani-tensor's N-D
functionality. That's a real v0.4.0 concern (matmul/dense layers), not
this phase's. Not declared as a dependency here; will be added when
v0.4.0 actually exercises it, matching this ecosystem's "only declare
what the current phase actually uses" convention.

Full verification: `vanic audit-safety` reports full `#[bounded_stack]`/
`#[wcet]` coverage (37 functions checked, 0 gaps) — `graph_forward`/
`graph_backward` are Vec-length-loop-bounded (a WCET formula comment,
not a literal attribute, same treatment as vani-symbolic's
Vec-length-loop-bounded functions) — the other 9 new functions carry
`vanic check`'s own exact reported worst-case for both attributes, per
MAINT-1 discipline. Full `vanic test` suite (all 6 test files, both
backends) and both examples (both backends) pass. Full SMT verification
(`vanic check` without `--no-verify`) hangs on this package — confirmed
**pre-existing**, not caused by this phase's code: an unmodified test
file (`tests/test_linreg.vani`) hangs identically. Same
verifier-performance class already found in vani-pde and
vani-probability this session, not a real defect.

## v0.4.0 — Layers, activations, losses ✅ implemented 2026-08-16

Depends on: v0.3.0.

- [x] Dense/linear layer — `graph_dense(arena, inputs, weights, bias)`,
      built by **composing existing `graph_mul`/`graph_add` calls**
      rather than a dedicated node kind (matmul emerges from scalar-op
      composition, matching how the losses below are also composed).
      **Not built via vani-matrix/vani-tensor** as originally scoped —
      this design's per-node scalar values never needed either, same
      scope correction v0.3.0 already made explicit.
- [x] Activations as graph node kinds: `graph_relu` (kind 7),
      `graph_sigmoid` (kind 8, reuses the `f64_sigmoid` builtin),
      `graph_tanh` (kind 9, reuses the `tanh` builtin). **`softmax`
      deliberately NOT added** — it's inherently multi-output (every
      output in a group depends on every input), a structural mismatch
      with this one-node-one-scalar design; this package's
      classification support has always been binary-only (see v0.1.0's
      `logreg_*`), so binary cross-entropy (below) covers the loss side
      without it. A documented scope narrowing, not silently dropped.
- [x] Losses: `graph_mse_loss` and `graph_binary_cross_entropy_loss`,
      both composed from existing primitives (`graph_sub`/`graph_mul`/
      `graph_add`/`graph_const`, plus a new `graph_log` node kind 10 for
      cross-entropy) — no dedicated "Loss" node kind, same composition
      philosophy as the dense layer.
- [x] Each new node kind validated with the same finite-difference check
      as v0.3.0 (`tests/test_graph_layers.vani`, 9 tests, cross-checked
      against `calculus::diff_central`): Relu on both its positive and
      negative branches separately (the kink at 0 makes a single
      straddling test meaningless), Sigmoid, Tanh, Log, an
      activation-composition test (`Relu(Sigmoid(x))`-shaped chain,
      proving the new kinds compose through the same backward machinery
      with no special-casing), a dense-layer test (forward value +
      per-weight gradient), an MSE test, and a binary-cross-entropy test
      (gradient matches the standard closed-form `sigmoid(z) - target`
      combined rule exactly — confirmed both analytically and via
      `examples/layers_demo.vani`'s worked hand-checkable example).

Full verification: `vanic audit-safety` reports full `#[bounded_stack]`/
`#[wcet]` coverage (45 functions checked, 0 gaps). Full `vanic test`
suite (all 8 test files, both backends) and all 3 examples (both
backends) pass. `examples/layers_demo.vani` (new file) walks through a
one-neuron dense->sigmoid->binary-cross-entropy chain end to end.

## v0.5.0 — Optimizers over the graph ✅ implemented 2026-08-16

Depends on: v0.3.0.

- [x] SGD — `graph_sgd_step`, stateless (`param -= lr * grad`).
- [x] Momentum — `MomentumState`/`momentum_state_new`/
      `graph_momentum_step` (`v = beta*v - lr*grad; param += v`).
- [x] Adam — `AdamState`/`adam_state_new`/`graph_adam_step`, bias-corrected
      first/second moment estimates, standard update rule.
- [x] Note: vani-optimize's existing solvers don't fit this signature
      (per-graph gradient vector, not `fn(ref Vec<f64>) -> f64`) — new code,
      same underlying math, not a straight reuse (confirmed during
      implementation, not just anticipated).
- [x] **Design note**: `MomentumState`/`AdamState` are indexed by
      *position in the caller's `params` Vec*, not by arena index — a
      caller training 3 params out of a 50-node arena gets a 3-element
      state, not a 50-element one. Each `*_step` function returns a
      FRESH state struct (`return MomentumState { velocity: new_velocity
      }`) rather than mutating one in place, matching this file's own
      dominant "construct once, return" convention
      (`KMeansResult`/`TrainTestSplit`/`KFoldSplit`) — also sidesteps
      needing direct struct-field-assignment syntax
      (`state.t = state.t + 1`), which has no precedent anywhere in this
      ecosystem's code to confirm works.
- [x] Validation (`tests/test_graph_optimizers.vani`, 6 tests): a
      hand-computed single-step exactness check per optimizer (using a
      FABRICATED `grads` Vec, isolating the optimizer's own arithmetic
      from autodiff correctness, already covered separately), plus a
      convergence check per optimizer (minimizing `f(x)=(x-3)^2` via
      `graph_mse_loss` over 200-300 iterations from `x=0`, the actual
      point of having an optimizer — not just correct single-step math).
      All 3 converge to within `1e-3` of the target on both backends;
      `examples/optimizer_demo.vani` (new file) runs all three side by
      side.

Full verification: `vanic audit-safety` reports full `#[bounded_stack]`/
`#[wcet]` coverage (50 functions checked, 0 gaps). Full `vanic test`
suite (all 9 test files, both backends) and all 4 examples (both
backends) pass.

## v0.6.0+ — Training-loop utilities ✅ implemented 2026-08-16 (optional/stretch)

Depends on: v0.1.0-v0.5.0.

- [x] Training-loop helper, batching — deliberately scoped down to
      `shuffled_indices(n, seed)`, the exact seeded Fisher-Yates
      permutation already duplicated inside `train_test_split`/
      `k_fold_split`, factored out as a standalone reusable utility (the
      existing two functions are left as-is, not retrofitted, to keep
      this phase's diff scoped to what it's actually adding). A
      genuinely GENERIC trainer function — one that takes a
      caller-supplied "build the graph" step — would need a
      ref-capturing closure to thread the caller's own data through, the
      same wall v0.1.0's `logreg_fit` and v0.3.0 already hit; not
      revisited here since nothing about this phase's own scope requires
      solving that problem.
- [x] **One** worked small-MLP example, deliberately narrowed from "a
      couple" — a 2-2-1 MLP (dense→sigmoid hidden layer, dense→sigmoid
      output, binary cross-entropy loss, Adam optimizer) trained on XOR,
      the canonical toy problem a single dense+sigmoid layer *provably*
      cannot solve (not linearly separable). All 9 trainable params
      (4 layer-1 weights, 2 layer-1 biases, 2 layer-2 weights, 1 layer-2
      bias) live in ONE arena and are reused across all 4 training
      examples' forward paths — real exercise of the DAG/shared-node
      gradient-accumulation property `tests/test_graph_core.vani`
      validated in isolation, this time for real (`graph_backward` must
      correctly SUM each param's gradient contribution across all 4
      examples). `tests/test_graph_training.vani` trains for 3000 Adam
      epochs and asserts every example is correctly classified
      (0.5-threshold, same convention as `logreg_predict`) plus the loss
      dropped to under 10% of its starting value.
      `examples/xor_mlp_demo.vani` runs the same training and prints
      before/after loss and predictions — loss goes from ~0.72 to
      ~0.0001, predictions land at ~0.0001/~0.9999/~0.9999/~0.0001
      against targets 0/1/1/0, identical on both backends. A second toy
      problem was considered and dropped: nothing it could validate that
      XOR doesn't already cover more convincingly.
- [x] Scope, not left open-ended: this phase is now considered CLOSED,
      not "revisit once used" — a real, passing, cross-backend-verified
      MLP training loop is the concrete "actually used" signal the
      original scope note was waiting for.

Full verification: `vanic audit-safety` reports full `#[bounded_stack]`/
`#[wcet]` coverage (51 functions checked, 0 gaps). Full `vanic test`
suite (all 10 test files, both backends) and all 5 examples (both
backends) pass. This closes out vani-ml's entire originally-scoped
roadmap (v0.1.0 through v0.6.0+) in one session.

---

## Future / explicitly out of scope for now

- CNN/RNN-specific layers — not scoped; revisit only if a real use case shows up
- GPU/SIMD-accelerated ops — depends on vāṇी compiler SIMD support maturity,
  tracked separately in vani-compiler's docs
