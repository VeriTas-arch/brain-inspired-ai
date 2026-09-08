# GPM and SGP: paired single-seed protocol

Frozen before inspecting candidate scores, 2026-09-07. Test GPM and SGP with
seed 61001, using the existing Pong boundary, Breakout heads, Adam state and RNG
states from `outputs/policy_replay_v1/seed-61001/anchor.pt`. No repeated seeds,
parameter sweep, additional regularizer, replay loss, or network expansion.

The existing anchor manifest supplies the PPO and evaluation configuration:
524,288 new-task transitions, 8 asynchronous environments, compiled policy,
128-step rollouts, 4 epochs, minibatch 32, deterministic raw-score evaluation
every 131,072 transitions with 10 complete episodes per task. Preserve the
previous four branches as frozen historical comparators. Verify that their
PPO, collector, environment and evaluation source hashes remain identical.

## Method and adaptation boundaries

Use the 2,048 uint8 observations already stored in the anchor's training memory;
do not use its teacher logits or independent probe to estimate subspaces. For
each shared convolutional layer, sample 16 receptive-field patches per observation
with replacement using a private RNG (seed + 40000). Use all observations at the
shared fully connected layer. Accumulate uncentered float64 second moments, whose
eigenvectors and square-root eigenvalues give the left singular vectors and
singular values of the input matrix. Retain the minimum rank explaining 0.995
of squared singular-value energy. Use the same basis for both methods.

GPM uses unit importance. SGP uses equation 2 of Saha & Roy (2023):
`lambda_i = (alpha + 1) * sigma_i / (alpha * sigma_i + sigma_max)`, with alpha 25.
These threshold/alpha values follow that paper's Atari experiment, not tuning on
this project's exposed scores. The original paper used different games, a larger
network and 10 million steps per task; this is a controlled local adaptation.

Preserve the existing biases by including a constant-one input coordinate and
projecting the joint weight-and-bias displacement. This is an explicit adaptation
of the affine projection argument, not a claim to reproduce a bias-free network.
Apply projection AFTER Adam moment scaling, through optimizer pre/post-step hooks.
Moments and PPO gradient clipping retain their original semantics. Measure the
realized displacement to quantify float32 subtraction/addition error. Project all
shared affine layers, so both policy and value contributions are covered. New
task heads update normally; old heads must remain bitwise unchanged.

Only the transition from one learned task to the next is implemented and tested.
Do not claim to implement SGP importance accumulation over three or more tasks.
The bases/importance, rather than the old observations, are needed during training;
persisted checkpoints/probes remain separate scientific artifacts.

## Validation and interpretation

Before full training, test subspace extraction against direct SVD, actual Adam
displacement projection (including inherited moments), hard-projection preservation
of affine outputs, scaled attenuation, convolutional receptive-field extraction,
and integration with the maintained PPO minibatch loop. Run a short GPU smoke in
a separate output directory; smoke results are not learning-quality evidence.

Report per-layer rank, captured energy, orthonormality, basis storage, retained
update energy, and realized projection error. Verify matching first rollout hash,
exact transition checkpoints and unchanged old heads. No automatic rank-dependent
retuning: a high-rank basis is an outcome to report, not a reason to alter settings.

Compare old-task retention, new-task endpoint and normalized learning-curve AUC
separately, with policy KL, action agreement and costs. A candidate that improves
only retention while worsening new learning does not resolve the existing tradeoff.
Use the prior descriptive margins (3 raw points for retention and comparisons to
the independent reference, minimum new score 5) only as secondary descriptors;
passing them does not establish superiority to replay or fine-tuning. Preserve
negative results. Single-seed conclusions are descriptive for this run.

```bash
direnv exec . python scripts/run_experiments.py train subspace \
  --study-config experiments/subspace_v1.json --seed 61001 --device cuda \
  --log-dir logs/subspace_v1
```

References: [GPM](https://arxiv.org/abs/2103.09762),
[SGP](https://arxiv.org/abs/2302.01386), especially SGP Appendix E (Adam-GP).
