MVE: Protocol v1.0 (IMDb × DistilBERT × RFF-Kernel Attention)
Objective

Show that replacing dot-product similarity with an RFF kernel similarity in the last 2 Transformer layers yields at least one measurable win:

Accuracy / F1 (primary), and/or

memory + speed (secondary), and/or

attention coherence / robustness (supporting evidence)

This aligns with the “quantum utility” framing (near-term gains without claiming advantage). 

2505.23860v3

1) Model Intervention
Baseline

distilbert-base-uncased for sequence classification (IMDb, binary).

Intervention scope (exact)

Replace attention similarity in only layers L4 and L5 (0-indexed: last two layers in DistilBERT’s 6-layer encoder).

Keep everything else identical: same number of heads, same projections, same softmax attention pipeline.

Freeze policy (exact)

Freeze embeddings + layers L0–L3

Train:

layers L4–L5

classifier head

your kernel attention parameters (if any)

This isolates the effect to semantic composition, as you intended.

2) Kernel Attention Definition

You’re replacing the “ruler”:

Standard similarity
Sij=qi⊤kjdk
S
ij
	​

=
d
k
	​

	​

q
i
⊤
	​

k
j
	​

	​

Kernelized similarity (RFF approximation of RBF kernel)
k(x,y)≈ϕ(x)⊤ϕ(y)
k(x,y)≈ϕ(x)
⊤
ϕ(y)
ϕ(x)=2mcos⁡(Wx+b)
ϕ(x)=
m
2
	​

	​

cos(Wx+b)

W∈Rm×dk
W∈R
m×d
k
	​

, with entries sampled from 
N(0,σ−2)
N(0,σ
−2
)

b∼Uniform(0,2π)
b∼Uniform(0,2π)

m
m = number of random features (your key compute knob)

σ
σ = kernel bandwidth (your geometry knob)

Then:

Sij=ϕ(qi)⊤ϕ(kj)
S
ij
	​

=ϕ(q
i
	​

)
⊤
ϕ(k
j
	​

)

and

A=softmax(S)
A=softmax(S)
What to keep “honest”

Don’t claim entanglement. Call it “Hilbert-space overlap via kernel feature maps.”

Your “quantum-ready” story is: same API; swap 
ϕ(⋅)
ϕ(⋅) with a quantum feature map later. 

2501.15630v2

3) Implementation Plan (HuggingFace + PyTorch)
Minimal code changes

Subclass DistilBertSelfAttention or monkey-patch only the similarity calculation inside it.

Keep Q, K, V projections as-is.

Insert:

phi_q = rff(q) and phi_k = rff(k)

scores = phi_q @ phi_k.transpose(-1, -2) per head (batched)

RFF map placement

Apply RFF per attention head on 
dk=hidden_dim/n_heads
d
k
	​

=hidden_dim/n_heads. This avoids mixing head subspaces.

Parameterization choice (important)

For the MVE, do one of these:

MVE-safe (recommended): fixed random W, b

Sample once per run, set as buffers (not trained).

Pros: stable, minimal moving parts.

Lets you claim: “kernel geometry does the work.”

Optional upgrade: learned bandwidth σ only

Keep W,b fixed; learn only σ (or equivalently scale inputs).

Pros: small trainable knob; still controlled.

Avoid learning full W in the MVE unless you want this to become “just another learned projection.”

4) Training Setup (fast, reproducible)
Data

IMDb from HuggingFace datasets.

max_length: 256 (for speed)

then one robustness stress test at 512

Optimizer & schedule

AdamW

LR: 2e-5 (baseline & kernel)

weight_decay: 0.01

warmup_ratio: 0.06

epochs: 3 (IMDb usually converges fast with DistilBERT)

batch_size: 16 (or 32 if you have VRAM)

gradient_accumulation: as needed

Seed discipline

Run 3 seeds (e.g., 13/42/1234). Committees trust variance reporting.

5) Ablations (exactly the three that matter)

You only need:

A1: Random feature dimension m

m ∈ {64, 128, 256}
Expectation: accuracy improves then saturates; compute grows linearly.

A2: Bandwidth σ

σ ∈ {0.5, 1.0, 2.0} (or set by heuristic from q/k norms)
Expectation: too small → overly local; too large → near-linear dot product.

A3: Swap depth

Swap only L5 vs swap L4+L5
Expectation: L5-only may already show effect; L4+L5 strengthens it.

That’s it. Don’t add more.

6) Metrics: What you must report
Primary

Accuracy

F1 (binary)

Efficiency (must-have)

Peak GPU memory (torch.cuda.max_memory_allocated())

Step time (ms/step) or tokens/sec

“Global coherence” evidence (simple but persuasive)

Paper 1 claims more globally coherent attention maps; you can test a lightweight proxy. 

2501.15630v2

Metric: attention entropy per head

Compute entropy of attention distribution for each query token:

Hi=−∑jAijlog⁡(Aij)
H
i
	​

=−
j
∑
	​

A
ij
	​

log(A
ij
	​

)

Report mean entropy (lower can mean more focused; interpret carefully).

Metric: attention distance bias

Average attention mass as a function of token distance |i–j|.
If kernel attention increases long-range mass without hurting accuracy, that’s a nice story.

7) Robustness mini-test (optional but high impact)

On the IMDb test set (small subset is fine):

Word dropout 5% (remove random non-stopwords)

Synonym swap for a few adjectives (simple thesaurus)
Measure accuracy drop vs baseline.

If kernel attention drops less, you’ve got a strong “geometry captures semantics” argument.

8) Acceptance Criteria (what “success” looks like)

You only need one win, but define it up front:

✅ Win A: +0.5–1.5% accuracy/F1 vs baseline (avg over 3 seeds)
or
✅ Win B: same accuracy ±0.3% but ≥15% memory reduction at max_length 512
or
✅ Win C: same accuracy but better robustness / long-range attention behavior

9) Repo layout (committee-friendly)
qra_attention/
  qra_attention/
    kernels/
      rff.py
      quantum_stub.py
    attention/
      kernel_self_attention.py
    patching/
      patch_distilbert.py
  experiments/
    imdb_train.py
    imdb_eval.py
    metrics_attention.py
  notebooks/
    attention_maps.ipynb
  results/
    tables.csv
    plots/
  README.md
  LICENSE


In the README, open with:

What you replace (“the ruler”)

What you keep fixed (architecture control)

How to reproduce

10) The 3 biggest pitfalls (and how you avoid them)

Kernel blows up scores (softmax saturation)
Fix: normalize φ(q), φ(k) or scale scores by √m (try both; pick one and keep fixed).

You accidentally change too much
Fix: keep V path identical and only change similarity.

RFF randomness causes noisy results
Fix: fix W,b per run + use 3 seeds; report mean ± std.