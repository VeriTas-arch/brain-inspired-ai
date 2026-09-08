# Bounded policy replay: frozen qualification protocol

Frozen before observing this study's scores, 2026-09-07. Development qualification
seed: 61001. Subsequent replication seeds: 61011, 61012, 61013. No seed selection,
weight sweep, memory sweep, or automatic structural escalation. This is an
experimental candidate, not a reproduction of CLEAR or RECALL.

The executable settings are in `policy_replay_v1.json`. The runner stores the
settings, source SHA-256 hashes, hardware, and Git parent for each seed. Generated
artifacts live under `outputs/policy_replay_v1/seed-<seed>/`. Reusing an output
directory with different code or configuration is rejected. Completed branches
are reusable; interrupted branches restart from the saved boundary, not midway
through an environment episode.

## Estimand and comparison

Train Pong once with shared-backbone multi-head PPO. Admit it only if ten complete
deterministic evaluation episodes average at least 5 raw points. From this exact
backbone, new Breakout heads, Adam state, and PyTorch RNG state, compare plain PPO,
current EWC (lambda 0.4, unchanged 100-sample final-rollout Fisher), and actor KL
replay. The independent reference preserves Pong and trains a fresh Breakout
backbone, with matching new heads but fresh optimizer state. Its different
representation and parameter count are intentional; it is not an upper bound.

All branches use the same environment seed, Breakout transition budget, PPO
minibatches, update epochs, evaluation seeds and intervals. Branch trajectories
will diverge as their policies diverge. The maintained runtime supplies rollout
collection, independent per-environment GAE, and optimization. Compiled runtime
warmup is included in training time, and compiler state is reset between branches.
No throughput result is interpreted as learning quality.

Boundary scores are evaluated once and reused where both model and evaluation
seed are identical: Pong at step zero in every branch, Breakout at step zero in
the three shared branches, and Pong throughout the frozen independent reference.
History explicitly labels reuse. All changed models are evaluated anew. Shared
boundary evaluation cost is outside branch wall time, so wall time is execution
cost under this reuse policy, not a controlled evaluation-speed comparison.

## Candidate

At the Pong boundary, sample 32,768 transitions of the frozen stochastic final
policy using evaluation wrappers (full episodes, raw rewards). Randomly retain
2,048 uint8 stacked observations and float32 teacher logits. Only these retained
pairs and task IDs are the candidate's persistent memory. Its sampler uses a
private CPU generator. Each PPO minibatch adds one equally sized replay batch
with weight 1.0 and KL(teacher || student), temperature 1. Old actors and the
shared backbone receive gradients. No critic, return, advantage, PPO ratio, EWC,
or other mechanism is replayed. Sampling is task-uniform when more tasks exist.

This protocol permits a separate boundary collection budget, reported explicitly;
the candidate is not matched to fine-tuning on total environment interactions.
All four branches do have identical *new-task training* transition budgets.
The boundary stream may finish midway through an episode; it is a state-sampling
stream, not a scored evaluation episode. Teacher targets are from the final policy,
not earlier versions encountered while learning Pong.

A separate seed and stream produce a 1,024-state held-out policy probe. It is used
only for KL and action agreement, never for optimization or target refresh. Its
32,768 transitions and all score evaluations are diagnostic overhead common to
the comparison. The common anchor checkpoint is an experimental artifact, not a
teacher model required by the deployed replay candidate.

## Outcomes and gates

Report raw scores separately for each task at Breakout steps 0, 131072, 262144,
393216 and 524288, retaining all episode returns. Report Pong final-minus-boundary
score, Breakout final score, trapezoidal Breakout AUC divided by transition budget
(units: raw reward), held-out old-policy KL, action agreement, persistent replay
tensor bytes, training time, wall time, and peak allocated/reserved CUDA memory.
CUDA peaks include evaluator/probe work and setup; they are end-to-end process
measurements, not isolated optimizer workspace. CPU diagnostic artifacts are
common and excluded from candidate replay storage.

The candidate passes this budget-specific descriptive gate only if:

- the old-task admission gate passed;
- candidate and independent Breakout final means are each at least 5;
- Pong mean drops by no more than 3 points;
- Breakout final mean and normalized AUC are each within 3 points of the
  independent reference or better.

These are practical thresholds, not statistical noninferiority tests. A single
seed qualifies execution and gives a provisional result; subsequent seeds assess
whether the same fixed candidate repeats it. Preserve failures. An anchor failure
does not support a retention verdict. A candidate failure does not prove capacity
conflict and does not authorize tuning on these exposed results. Report paired
seed-level differences and episode variability without claiming population-level
certainty from three seeds.

## Commands

```bash
direnv exec . python scripts/run_experiments.py train policy-replay \
  --study-config experiments/policy_replay_v1.json --seed 61001 --device cuda \
  --log-dir logs/policy_replay_v1
direnv exec . python scripts/run_experiments.py train policy-replay \
  --study-config experiments/policy_replay_v1.json --seeds 61011 61012 61013 \
  --device cuda --log-dir logs/policy_replay_v1
```

The study config is the authority for protocol options; use generic CLI device,
seed, log and scheduling options only for this suite.

## Scope amendment: 2026-09-07

After completion of seed 61001, the user requested no multi-seed runs. Replications
61011, 61012 and 61013 were stopped and their incomplete artifacts retained but
excluded from scientific results. The original protocol above remains as a record
of the initial plan; its replication command is no longer a pending task. Final
interpretation is descriptive and limited to the completed single-seed comparison.
No candidate parameters or outcome thresholds were changed.
