# vani-ml

Machine learning library for the [vāṇी compiler](https://github.com/enthusiasticgeek/vani-compiler).
Staged: classical ML first, an autodiff/neural-net engine on top. Full scope, phase
breakdown, and risk notes live in
[kosh-index/ROADMAP.md](https://github.com/enthusiasticgeek/kosh-index/blob/main/ROADMAP.md#planned-ml-tier-scoped-2026-07-25) --
this README tracks only what's actually implemented.

**Status (2026-07-25): scaffolded, no functions implemented yet.** See TODO.md for
the phase checklist.

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

## Design note: autodiff without ref-capturing closures

vāṇी's closures support move/Copy captures (shipped 2026-07-15) but not
ref-capturing closures -- that's compiler path-D, deferred indefinitely (see
`vani-compiler/docs/missing_features.md`, "Lifetime variables"). The autodiff core
(v0.3.0) does not need it: it uses the same flat-arena pattern `vani-symbolic`
already uses for its expression tree (`Vec<Node>` + `i64` child indices), with
gradient buffers threaded through as explicit `mut ref Vec<f64>` arguments instead
of captured. See ROADMAP.md for the full writeup of why this was resolved without
a compiler change.

## What's included so far

Nothing yet -- v0.1.0 is scoped, not implemented. See TODO.md.

## License

MIT
