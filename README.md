# vani-ml

Machine learning library for the [vāṇी compiler](https://github.com/enthusiasticgeek/vani-compiler).
Staged: classical ML first, an autodiff/neural-net engine on top. Full scope, phase
breakdown, and risk notes live in
[kosh-index/ROADMAP.md](https://github.com/enthusiasticgeek/kosh-index/blob/main/ROADMAP.md#planned-ml-tier-scoped-2026-07-25) --
this README tracks only what's actually implemented.

**Status (2026-08-16): v0.1.0-v0.3.0 implemented and passing** (6 test
files under `vanic test`, LLVM and C backends both verified on both
examples, `vanic audit-safety` reports full `#[bounded_stack]`/`#[wcet]`
coverage). Full SMT verification (`vanic check` without `--no-verify`)
is slow/hangs on this package -- a known, pre-existing compiler
verifier-performance characteristic unrelated to this package's code
(see TODO.md's v0.3.0 closure writeup); use `VANIC_NO_VERIFY=1` for fast
iteration. See TODO.md for the phase checklist.

## Why classical ML first

v0.1.0-v0.2.0 are mostly glue over already-published packages (`vani-probability`'s
regression, `vani-optimize`'s gradient descent) -- low risk, ships fast, and proves
out the repo/CI/publishing mechanics before the harder work. The autodiff engine
(v0.3.0+) is a new architecture and is deliberately sequenced after that, not before.

## Why one repo, not several

This ecosystem draws repo boundaries by *domain* (`vani-tensor` vs `vani-matrix`
because N-D arrays and dense 2D linear algebra are different domains), not by
architectural layer. Classical ML and neural nets are one domain (machine learning)
at two capability levels -- staged as versions in one repo, the same way
`vani-symbolic` stages construction -> simplification -> differentiation.

## Add to your project (once published)

```toml
# vani.toml
[deps]
ml = { registry = "kosh", version = "^0.1" }
```

```sh
vanic add ml
vanic build
```

## Design note: no ref-capturing closures -- and it already mattered in v0.1.0

vāṇी's closures support move/Copy captures (shipped 2026-07-15) but not
ref-capturing closures -- that's compiler path-D, deferred indefinitely (see
`vani-compiler/docs/missing_features.md`, "Lifetime variables"). This wasn't
just a v0.3.0-autodiff concern: it already bit `logreg_fit` in v0.1.0.
`vani-optimize`'s `gradient_descent_fixed`/`gradient_descent_backtracking`
take fixed `fn(ref Vec<f64>, i64) -> f64` objective/gradient function
pointers -- there's no way to thread the training data (`X_flat`, `y`)
through one of those without a ref-capturing closure. So `logreg_fit` has
its own small dedicated gradient-descent loop instead, same algorithm
shape, specialized directly to `(X_flat, y, beta)`. The autodiff core
(v0.3.0) will hit the same wall in a bigger way and uses the same kind of
workaround: the flat-arena pattern `vani-symbolic` uses for its expression
tree (`Vec<Node>` + `i64` child indices), gradient buffers threaded through
as explicit `mut ref Vec<f64>` arguments instead of captured. See
ROADMAP.md for the full writeup of why this was resolved without a
compiler change.

## What's included (v0.1.0-v0.3.0)

| Module | Functions |
|---|---|
| Linear regression | `linreg_add_intercept`, `linreg_fit`, `linreg_predict`, `linreg_r_squared` -- thin wrappers over `probability::mlr_*` |
| Logistic regression | `logreg_fit` (own gradient-descent loop, see design note above), `logreg_predict_proba`, `logreg_predict` |
| K-means clustering | `kmeans_fit`, `kmeans_predict`, `kmeans_inertia`, `KMeansResult` -- Lloyd's algorithm, random-point init, genuinely new code (no existing kosh package covers this) |
| Train/test split | `train_test_split`, `TrainTestSplit` -- seeded Fisher-Yates shuffle |
| Core metrics | `mse`, `accuracy`, `precision`, `recall`, `f1_score` (binary labels `{0.0, 1.0}`) |
| Data utilities (v0.2.0) | `StandardScaler`/`standardize_fit`/`standardize_apply`, `MinMaxScaler`/`normalize_fit`/`normalize_apply`, `one_hot_encode`, `Dataset`, `KFoldSplit`/`k_fold_split` |
| Autodiff core (v0.3.0) | `GraphNode`, `graph_arena_new`, `graph_const`/`graph_param`, `graph_add`/`graph_sub`/`graph_mul`/`graph_div`/`graph_neg`, `graph_set_value`, `graph_forward`, `graph_backward` -- flat-arena reverse-mode automatic differentiation, no recursion (see TODO.md's v0.3.0 writeup for the full design) |

All `X_flat` arguments are row-major `n_obs x n_pred` (or `n_obs x n_dim` for
k-means), matching `vani-matrix`/`vani-probability`/`vani-tensor`'s shared layout.
See `examples/ml_demo.vani` and `examples/autodiff_demo.vani` for end-to-end tours.

## Known upstream issue found while building this

While writing `tests/test_logreg.vani`, a standalone unary-minus float literal
(e.g. `-3.0` as a `vec()` argument) panicked the `vanic` LLVM backend at codegen,
even though `vanic check` accepts it cleanly. Filed upstream as BUG-6 in
`vani-compiler/docs/TODO_CURRENT.md`, not fixed. Worked around throughout this
repo by writing `0.0 - 3.0` instead of `-3.0`, consistent with this ecosystem's
existing "no unary minus" convention (see `docs/reference_vani_language_notes.md`-style
notes in the wider project).

## License

MIT
