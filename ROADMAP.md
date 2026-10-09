# Research roadmap — Epigenetic language models (Free-Slot Adaptation)

Synthesized from two independent AI-assisted adversarial brainstorms (2026-10-08), each given the full result set. They converged almost completely; where they diverge is noted.

## The honest status (both reviewers, unprompted)

> What we have today is **dense PackNet on a random 50% mask in a fact-memorizing MLP.** `pair8` *never enters the adaptation loop* — `src/freeslot.py` trains fp32 weights. The zero-forgetting result is an **invariant of freezing, true by construction** (retention 1.000 is a unit test that the mask was applied — it holds even on a model that learned nothing), **not a theorem or a finding.** The capacity cliff and the LoRA cost comparison are real but *local to the toy*.

So the current results are a **mechanism demonstration**. Whether FSA *exists for the actual ternary format and on a real model* is untested — and that is the whole ballgame. Everything below is gated on answering that, cheaply, before any scale-up.

## Flagship framing: a living personal model ("YOUR AI")

The reviewers judged FSA modest for *general* continual learning (LoRA is more capable; the storage win is conditional; the hard variant needs task-ID). **Those same facts become advantages for one specific product: a private, on-device assistant that learns continually about a single user.** This is the north star, and it should steer the experiments.

**What survives adversarial review (both reviewers), stated honestly:**
- The surviving unit is a **revocable domain/skill patch**, not per-fact memory. *Patch-revocation contract:* deleting a patch removes its parameter contribution exactly; it does **not** delete individual facts (facts in a slice are superimposed) and is only counterfactual if no retained patch depended on it.
- **Masked-base zero-forgetting** holds only with the patch *off*; the *enabled* personalized forward does change (unmasked retention is negative for hard FSA; the soft always-on model degrades 0.98→0.08 with load).
- **RAG/OPPU is the baseline to beat, and for arbitrary facts it wins** — a retrieval row (or a per-user adapter) deletes per-fact, leaves the base bit-identical, needs no task-ID. The honest case for weights is **skills / style / policy** that must act in every forward pass, not facts.
- "No params/FLOPs" is about *inference*; on-device **training** is unestablished (AdamW state on a half-tensor reserve ≈ 10 GB at a few-B weights).

**So the sharpened framing:** the thing worth building is a **private, revocable skill/policy layer** in a quantized model — and the right evaluation pits it against **OPPU-style per-user adapters + local retrieval** on a real personalization benchmark (LaMP), with unlearning measured (SISA/TOFU). This is still appealing, but it is a hypothesis gated on Gate 0 — whose storage outcome is **genuinely open** (the format assay can't forecast a frozen-base reserve; it could go either way).

## The gate — do these first, stop at the first failure

### Gate 0 — Ternary reserve vs fp32 reserve (days, can kill the project cleanly)
The assay gives every free slot a full **fp32** value. Production pair8 gives it a **trit + a shared block scale**, placed by SR-STE while `S₀` is frozen. *A fp32 reserve holding ~460 kbits does not imply a ternary reserve holds anything useful.*

**Do:** same 3-layer MLP, same saturate-then-adapt, but quantize **both** `S₀` and `Z₀` to pair8 (ternary, ≤2-of-4 pairs, block scale) in training and at eval. Phase 1 trains base pairs; phase 2 freezes those pairs **and their scales** and trains only reserve pairs under the format constraint. Report `K_B` and the cliff next to the fp32 curve, plus a **ternary-LoRA of matched added bytes** (not matched rank).

**Go:** ternary `K_B` ≥ ~½ the fp32 curve. **No-go:** ternary `K_B` collapses toward the `frozen_noslots` control ⇒ FSA does not exist for this format; the dense-mask results stay a mechanism note. *(This is also the fix for the fp32-harness gap — building it is how we test the real claim.)*

### Gate 1 — One real transformer: facts vs domain vs skill, byte-matched (the MVP, ~1 week)
Random isotropic keys are the **best case for a high-rank sparse write and the worst case for LoRA** — the toy flatters us. Real post-training updates are low intrinsic rank; structural zeros are exactly the coordinates the quantizer refused to spend a trit on.

**Do:** one pretrained decoder (**Pythia-410M** for a 1-GPU week, or **Qwen2.5-1.5B / Llama-3.2-1B**). Impose the pair pattern **by magnitude** within each block of 8 on gate/up/down only (keep the 2 largest-magnitude pairs as `S₀`); heal the kept weights briefly with `Z₀` forced to zero; freeze embeddings, attention, norms, biases, `S₀`. Three targets at matched **bytes**: (i) ~5k entity facts the base gets wrong, (ii) code / rare-language next-token loss, (iii) a narrow skill. **Baselines:** QLoRA r=8/16 on the same projections, **a new expert / extra MLP of equal bytes**, full-FT (forgetting control), frozen-`Z₀` (no-capacity control).

**Report six numbers and stop:** masked base ppl before/after (Δ≠0 is a bug), unmasked base ppl, fact exact-match, domain ppl, added bytes, ternary-`K` vs fp32-`K`. **Metric is bytes/token, not FLOPs** — decentralized MoE decode is bandwidth-bound; a filled slot moves bytes.

**Go:** fact metric ≥ 80% of QLoRA-r8 at ≤ 4× added bytes, masked base loss unchanged. **Expected (both reviewers):** sharp specialization — facts work but several× less byte-efficient than r=8; **domain loss barely moves** (norms/router frozen); a reasoning skill doesn't fit; a new expert wins domain specialization. If that's the pattern, the honest paper is *"isolated fact-packing + exact revoke,"* not *"continual learning."*

## The big swing (only if the gates pass)

**Make the quantization format a capability sandbox with CRDT-merge semantics for a permissionless ternary MoE.**

Contributors don't ship adapters and don't average gradients. They receive, **by hash**, an index set inside `Z₀`; train locally against the frozen packed base they already store; publish only filled trits. Any node merges by **union**. The base forward (sidecar off) is **identical no matter who wrote**. **Revocation = deleting an index set** (exact, even after merge). A domain **router** turns bundles on; the base stays one GEMM.

This is bigger than continual learning — adapters already learn the task. What adapters *can't* give on decentralized hardware: **untrusted writes that can't clobber the base or each other, exact takedown/unlearning, and no second kernel.** PackNet never had it (its free weights weren't a packed deployed format); a merged low-rank update has no disjoint support to delete. **It matters only if Gate 1 shows a useful amount of real knowledge fits at a byte premium worth paying for isolation** — if the premium is ~10× and contributors are trusted, ship QLoRA or a new expert instead.

**Experiment:** on the healed small LM, hash-allocate the reserve (domain *d* owns a pseudorandom block/pair subset ⇒ **mask storage ≈ 0**, the only regime where FSA is storage-competitive). Train 8–32 domains; merge by union; measure base-after-merge (masked), each domain (`S₀∪R_d`), the **collision tax**, **exact revoke** (deleted domain → chance, others unchanged), and a **router** (centroid of domain embeddings, or a tiny classifier on the frozen residual stream) picking top-1/top-2 bundles with no task-ID.

## Supporting directions

- **ρ Pareto, priced in bytes and base quality.** ρ=0.5 is inherited from the pair8 rule, not an optimum; the toy's ~18% base-bit cost won't survive a model trained near its limit. Sweep ρ∈{0,0.05,0.1,0.25,0.5}; put *pair8 ρ=0 + quantized LoRA* and *smaller base + new expert of size ρN* on the same plot. **"Compression and tenant capacity are one dial" is the real scientific claim — currently a sentence; the sweep is the claim.**
- **Sidecar kernel, not in-place densification.** Deployed object = frozen pair8 base pack **+ a sidecar** of `(block id, pair index, trit, scale override)` the kernel applies or skips. Resolves the codebook-exit problem *and* the "0 FLOPs" caveat, and forces the structural hash allocation that keeps the mask from dominating. Benchmark base vs +1% vs +50% sidecar vs byte-matched LoRA at real expert scale; publish wall-clock and bytes/token.
- **Function-space interference of simultaneous bundles.** Disjoint supports ⇒ **parameter** interference is exactly zero; **functional** interference (two sidecars live at once) is not — our unmasked retention of −42 at 500 facts is the warning. This, not another merging baseline, is the open question vs TIES/DARE.
- **Exact unlearning / revoke as a main result** (not an appendix) — the rare property; may matter more than acquisition for poison, takedown, bad contributors.
- **Negative controls:** (a) interpretability — ablating `R_d` must remove the new behavior and leave the base; if it survives, the reserve learned nothing. (b) MEMIT-style **override probe** — FSA can *add* but cannot *cancel* a base weight in `S₀`; if it can't flip a base fact, say **"addition-only"** in the abstract and don't cite editing as the comparison.

## Positioning (both reviewers agree)

> **A layout-and-isolation result for extreme-quantization MoE, not a continual-learning algorithm.** The algorithm is PackNet; the *object* is a pre-addressed write buffer inside the packed expert, with compression as the uncommitted buffer.

Lead the paper with the **real-model cost frontier** and the **contribution/isolation architecture**; the synthetic assay is supporting analysis. **What's genuinely new (if demonstrated):** (1) the reserve is priced into the format *before any task exists* — ρ, base quality, tenant capacity, and file size are one allocation choice; (2) **disjoint ternary commits are a merge algebra** (union, exact revoke, confinement of a bad writer); (3) the cliff is a real **operating limit / write-refusal trigger** for a permissionless system. **What's over-claimed and must be toned down:** the "zero-forgetting theorem," headline retention 1.000, the soft middle-ground (LwF-with-rehearsal, a small-load trick), and anything that reads as a *capability* win over LoRA (it's cost structure).

**Cite & differentiate in the first two pages:** PackNet, Piggyback, HAT, Composable Sparse FT (Ansell et al.), Diff Pruning, SPDF, Ada-QPacknet, TinySubNets, LwF, QLoRA, DARE/TIES, MEMIT, GRACE, Branch-Train-Merge, Sparse Upcycling, BitNet b1.58, 2:4 sparse tensor cores.

## Where the two reviews diverged
- **κ interpretation:** the *format* assay and the *adaptation* assay are different numbers; **pair8 is a slightly worse memory than plain ternary** — the aligned zeros (FSA) are the *only* reason to keep pair8's structure. Report them separately.
- **"Frozen base can't refine":** one review says it's too strong (reserve connections *can* reshape representations / suppress outputs); the other says it's **addition-only** (can't cancel an `S₀` weight). Reconciled: the reserve adds new paths that can suppress/augment outputs, but cannot edit a base weight — state precisely, and settle it with the override probe.
- **Soft middle-ground:** one review treats it as a direction; the other says it's a proven small-load trick, don't sweep λ further — **the router is the task-free version worth building.** (Our λ data agrees.)

## Accuracy corrections being folded into README/paper
1. Current results are **dense fp32 + mask (mechanism demo)**; pair8/ternary is **not** in the adaptation loop yet.
2. Filling zeros can push a block **past ≤2-of-4 → leaves the pair8 codebook**; patches need the **sidecar**, so "1.088 bits/weight for patches" is not automatic.
3. **Freeze the block scales too** (+2 bits/weight for an FP16 scale per 8 weights, to be charged).
4. "Zero extra FLOPs" is **kernel-conditional**.
5. Zero-forgetting is **true-by-construction**, de-headlined.
6. The capacity ceiling may be partly **optimization**, and is an **operating limit**, not just a curve.
7. κ reported as **two separate assays**; pair8 ≤ plain ternary as a memory.
