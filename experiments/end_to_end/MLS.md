# Masked Gauss–Newton ternary steps (C5 / v28)

Select `--method fsa_forced --optim mls --anchor PATH`. Both up and down
projections participate, or use `--down-only`. Dense and `--sparse-state`
work, including a thinned sparse reserve. Codes remain exactly
`{-1, 0, +1} * alpha_row` on the forced slot mask. Export, reload, TRAJ
evaluation and exact revoke use the existing harness paths.

Each iteration records actual expert inputs and linear-output gradients
from fact CE, including autograd's routing weights. Saturated facts, when
`--saturation-sampling` is enabled, retain their forwards but contribute no
CE gradient. The acceptance pass reuses that iteration's sampled tokens,
loss mask, denominator and saturation decisions without updating the sampler.

The row-batched solve uses eight masked CG iterations by default:

`H u = -eta X_f.T G`, with
`H = X_f.T X_f + lambda_g (diag(D) + U U.T) + gamma I`.

`D` and `U` refresh from 32 training anchor windows every 50 steps. A
one-pass randomized covariance range with eight oversampling directions
feeds an SVD; its leading vectors times singular values form the rank-16
`U`. Sketches stay in bf16 on the model device. No raw general activations,
full covariance, or per-row Gram matrices are retained. One expert's dense
solve workspace is temporary; sparse state uses `idx` to gather/scatter slots.
Candidate storage is bounded by the live-code budget plus incumbents, or
by the change cap when no budget is supplied.

The continuous proposal is `w = alpha*q_current + u`, so existing codes
can die or flip. The largest half of eligible coordinates per row are
rounded first; their residual drives two CG correction iterations for the
remaining coordinates. `--mls-thr` defaults to 0.5. Global `--budget` keeps
the largest continuous `|w|/alpha` coordinates. MLS support means fixed
eligibility; its budget caps nonzero codes rather than vacant support slots.
At most `--mls-max-change 50000` changes are selected by predicted finite
coordinate gain. Excess births are dropped if the change cap omits deaths
needed for a budgeted exchange.

`--mls-eta 0` (default) recalibrates every iteration to
`--mls-target-changes 20000` discrete changes before residual correction.
After a unit-eta solve `v`, each vacant coordinate reaches a rounding
boundary at `eta = threshold / (|v|/alpha)`. An existing code moving toward
zero first changes at `eta = (1-threshold) / (|v|/alpha)`; it can die or,
at larger displacement, flip, but counts only once. Reinforcing an existing
code contributes no crossing. Eta moves just beyond the globally Nth earliest
boundary, with ties split deterministically in module/slot order. Calibration
retains only O(N) crossing times and solves one linear at a time. If fewer
than N coordinates can change, all available crossings are used; zero
directions produce an empty transaction. Automatic calibration requires
`0 < --mls-thr < 1`.

Rejects halve N with a floor of 500 (or the initial N when smaller); accepts
multiply N by 1.25, capped at its initial value. Residual correction and the
existing budget and change cap can alter the final change count. An explicit
positive `--mls-eta` skips discrete calibration and keeps the existing eta
adaptation: rejects halve eta; accepts multiply it by 1.5 up to its initial
value. `--mls-trust` remains accepted for compatibility and is ignored.

Acceptance requires strictly lower fact loss plus the existing relative
local-output penalty, evaluated on the same two sampled anchor windows
before and after. Its weight is `--local-lambda` when positive, otherwise
`--mls-gen-lambda` (default 1). Non-finite or worse trials restore exact
master values; evaluation exceptions also roll back. `MLS` JSON lines and
the result's `mls.iterations` record predicted/actual trial changes,
births/deaths/flips, acceptance, eta, N and next N, and the change count before
residual correction. Each line also reports global `|u|/alpha` quantiles
(50/90/99/99.9/max) over eligible coordinates with fact gradients, before
residual correction. Percentiles use a streaming log histogram (about 1.1%
relative bin error); max is exact. Automatic mode reports next eta as null
because the next direction must be solved before eta is known. Predictions are local
quadratic surrogates; actual changes include the nonlinear full model and
measured local preservation penalty.

Expert checkpointing is suspended during fact recording to avoid replay
duplicates, then restored. Rehearsal, teacher distillation, binding loss,
drift/gate penalties, dual local targets, and Bop support/chaos strategies
are rejected with MLS. Other optimizers retain their existing defaults.

CPU checks: `python -m pytest -q experiments/end_to_end/tests/test_mls.py`.
