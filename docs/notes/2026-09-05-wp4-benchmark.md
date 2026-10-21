# 2026-09-05 - WP4, measuring the thing rather than describing it

WP4 of ADR-0002. The README already quoted numbers - 0.366 s for nine people,
2.45 fps end to end, 5.5 GB of Metal driver memory - and none of them could be
reproduced from the repository. They came from runs whose configuration,
machine and code state were not written down anywhere. This work package does
not improve any of those numbers; it makes the next ones traceable.

## Shape

`experiments/benchmark/*.toml` names an experiment and says how to time it:
warm-up frames, measured frames, repetitions, and whether the output is a
skeleton or a mesh. `unproject benchmark` writes one raw result per invocation,
`unproject benchmark-summary` folds every result into `summary.json` and
`summary.csv` without touching the model.

The runner opens the same `MultiHmrEstimator` the bodies stage opens and calls
the same `session.frame`. That is deliberate: a benchmark with its own
inference path measures the benchmark.

## What the mesh factor actually costs

Both output modes run identical inference, so measuring them the same way would
have made the RQ2 output factor meaningless. Mesh mode additionally builds each
track's unposed body the first time that person is seen, through the same
session calls the socket uses - once per track, not once per frame, because
once per track is what the viewer pays.

## The dry run nearly poisoned the summary

`--dry-run` exists so CI can exercise the orchestration without 1.6 GB of
weights. Writing it, I ran the dry run and the real run of the same benchmark
back to back and the summary averaged them: two microseconds from a stand-in
folded in with half a second from the model. The result file gives no hint -
same schema, same fields, plausible numbers.

`dry_run` is now a recorded field on the result rather than something inferred
from the checkpoint digest, and aggregation drops those results and says how
many it dropped. The two contaminated files were deleted; they were minutes old
and measured nothing.

This is the failure mode ADR-0002 §3.3 is about. It would not have produced an
error, only a wrong number, and the wrong number would have looked fine.

## `validate-config` had to learn a second schema

`experiments/` now holds two kinds of config, and the ADR §11 gate walks the
whole directory. They are told apart by content - a benchmark names the
experiment it measures - rather than by which folder they sit in.

## First real measurement

On the fixture clip, which is a 320x240 synthetic pattern with nobody in it and
therefore not a thesis number:

```
Apple M3 Pro, mps, 16 steady-state frames over 2 repetitions
median 0.5254 s per frame (1.90 fps)
cold start 10.22 s, reported apart
```

It exists to show the path works end to end against the real estimator. Real
cells wait on the frozen protocol, which is WP6.

## Not done here

- No cell of the RQ2 matrix is filled in. The two fixture configs are
  orchestration checks, and `docs/experiments/protocol.md` is still a draft.
- The live transport overhead - the 2.45 against 2.60 fps in the README - is
  not measured here. It needs the socket, and ADR-0002 §12 says the benchmark
  must not require the browser except for an explicit end-to-end test.
