# Simulation scale and performance

Path counts are configuration choices, not measures of model quality. More paths reduce Monte Carlo noise but consume more CPU time. The appropriate count depends on the engine and the precision needed from its output.

## Core API defaults

The exported APIs deliberately have different defaults:

**Static posterior season simulation — 50,000 paths.**

`simulate_seasons_from_posterior` inherits `league_table.SIMULATIONS`, whose value is 50,000. The exact-covariance implementation processes these paths in vectorized batches of at most 5,000.

**Chronological league simulation — one outer path.**

`simulate_seasons_chronologically` defaults to one outer path, 500 inner motivation paths, and one worker. One outer path is suitable for low-level execution and tests, not stable season probabilities.

An **outer path** is one possible future season: it samples scores, advances the calendar, updates team strength, and produces one final table. An **inner motivation path** is a conditional remaining-season simulation used only when estimating table-objective leverage for a relevant match date inside an outer path.

Do not estimate work as `outer paths × inner paths` and assume every product term is a complete season. Inner work is performed only where motivation scoring is active and depends on the path's current date, table, remaining fixtures, and loaded policy gates.

## Operational scale in the larger application

The private application's nightly command uses these defaults when simulation flags are omitted:

- 10,000 outer joint league-and-Cup paths;
- 500 inner motivation paths where motivation evaluation is required; and
- `min(16, os.cpu_count())` process workers.

The exact-covariance joint-calendar worker groups at most 256 outer seeds into one vectorized block. The number of blocks depends on path count and worker count.

This operational configuration is documented for scale, but the joint-calendar orchestrator and its input assembly are not exported. The export's chronological API therefore still defaults to one outer path and one worker.

The larger application also has a live-snapshot helper whose own default is 10 outer paths. That helper is not exported, and its small default is for development. The nightly command passes its 10,000-path default into the helper, so the helper's value must not be mistaken for production scale.

Historical frontend calculations use separate policies:

- historical league snapshots use 10,000 static paths;
- historical Cup projections use 1,000 paths; and
- vectorized posterior code processes at most 5,000 paths per batch.

These are application choices rather than hidden changes to the core API defaults.

## Measured local benchmark

A real-world implementation benchmark ran on one local 16-thread Windows machine with Python 3.14.7. It used cached inputs, isolated data copies, 10,000 outer paths, 500 inner motivation paths, and 16 workers.

**Parent implementation:** 131.634 seconds.

**Python batching plus narrow native ranking:** 87.705 seconds.

**Full native conditioned simulator:** 65.350 seconds.

The final treatment used 50.4% less elapsed time than the parent, equivalent to about a 2.01× end-to-end speedup. The 50.4% figure is the elapsed-time reduction for the complete measured run, **not** the speedup of native code in isolation. The native extension was active at both treatment checkpoints.

For a fixed 500-path comparison, the maximum numeric difference from the parent was \(4.26\times10^{-14}\), with no nonnumeric differences. This checks numerical consistency for that comparison; it is not predictive validation.

## How to interpret the timing

These are single-run contextual measurements, not a runtime guarantee. Hardware, operating system, remaining schedule, active motivation dates, process startup, compiler output, and other machine load can all change elapsed time.

The benchmark shows that batching and the optional native conditioned simulator can materially reduce one representative nightly build. It does not imply that every 10,000-path run will finish in about 65 seconds, or that increasing simulation count improves the statistical model itself.

## Choosing a count

Use a small count to test wiring, then increase it until the probabilities relevant to your use are stable enough. Keep the seed and all model inputs fixed when comparing counts or implementations. Report outer and inner counts separately, along with worker count and whether the native extension was active.
