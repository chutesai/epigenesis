# Serving cost of FSA patches in the packed pair8 runtime

**Question.** The main README says an FSA patch costs nothing extra at inference, because the filled slots sit
inside pair8 blocks that the base kernel already multiplies. This experiment tests that claim on the released
lambda 8B GGUF (`parallax-lambda-790B-t9.gguf`, sha256 `489cd1d2…`) with the `llama-parallax` runtime, on an
Apple M5 Max (CPU and Metal) and on a Snapdragon 8 Elite phone.

**Answer.** Within measurement noise, decode speed, prompt-processing speed, peak memory and file size are the same
for the base model and for patches of 670k and 3M slots. The largest paired difference in any median was 1.8%,
smaller than the run-to-run spread. The kernel does the same work for every legal block, so no difference is
expected. The patches are synthetic (see Limits); learned patches from the training harness have not been
benchmarked yet.

## Results

Medians over all samples, with [min, max] in brackets. The "paired" column is the median, over rounds, of
patched/base measured in the same round (model order rotated each round). 670k slots matches the size of the earlier Bop patches (the current best patch has 250k slots). 3M is a stress test, 4.5× larger.

**Apple M5 Max, decode and prompt processing (tok/s)**

| backend / kernel | model | decode | paired decode ratio | prefill (512 tok) | paired prefill ratio |
|---|---|---:|---:|---:|---:|
| Metal `lut9` | base | 174.8 [165.6, 177.5] | — | 1075 [1059, 1082] | — |
| | +670k slots | 172.3 [166.2, 177.5] | 0.999 [0.960, 1.006] | 1075 [1059, 1082] | 0.999 [0.997, 1.001] |
| | +3M slots | 175.1 [166.1, 177.6] | 0.999 [0.973, 1.032] | 1076 [1056, 1082] | 1.001 [0.997, 1.003] |
| CPU `lut8`, 8 thr | base | 102.2 [98.4, 104.2] | — | 371 [361, 388] | — |
| | +670k slots | 102.0 [96.1, 104.4] | 1.005 [0.975, 1.022] | 374 [366, 388] | 1.002 [0.974, 1.045] |
| | +3M slots | 101.2 [97.8, 103.7] | 0.990 [0.976, 1.004] | 369 [365, 386] | 1.009 [0.957, 1.021] |
| CPU `lut`, 8 thr | base | 66.0 [64.4, 67.0] | — | 148.0 [145.7, 149.1] | — |
| | +670k slots | 66.1 [63.6, 67.1] | 1.006 [0.984, 1.015] | 148.0 [144.8, 149.0] | 1.004 [0.985, 1.007] |
| | +3M slots | 66.0 [64.8, 67.1] | 0.998 [0.989, 1.013] | 148.4 [147.6, 149.0] | 0.999 [0.996, 1.012] |

Random-token benchmark (`--benchmark 128 --benchmark-depth 512 --prefill-batch 128 --repetitions 3`, fixed seed),
5 processes per cell, 15 samples. A second run on a fixed 512-token held-out passage with 128 greedy tokens gave
the same picture: paired decode ratios 0.982 to 1.000, prefill 0.998 to 1.005. All three models produced the same
128 greedy tokens there.

**Memory, size, load (M5 Max)**

| | base | +670k | +3M |
|---|---:|---:|---:|
| GGUF file size (bytes) | 2,133,006,592 | 2,133,006,592 | 2,133,006,592 |
| peak RSS, Metal `lut9`, 512+128 tokens | 2.752 GiB | 2.752 GiB | 2.752 GiB |
| peak RSS, CPU `lut8`, 512+128 tokens | 3.289 GiB | 3.289 GiB | 3.289 GiB |
| process start → first token, Metal (s) | 2.06 | 2.06 | 2.07 |

**Snapdragon 8 Elite (SM8750) phone, Q8_0-trunk t9 GGUF, CPU `lut8`, 6 threads pinned**

| model | decode tok/s | paired ratio | prefill tok/s (512) | paired ratio | peak RSS (short-prompt run) |
|---|---:|---:|---:|---:|---:|
| base | 57.0 [54.2, 58.4] | — | 209 [195, 226] | — | 3.244 GB |
| +670k slots | 56.8 [54.7, 58.7] | 1.002 [0.974, 1.034] | 200 [195, 239] | 0.996 [0.910, 1.053] | 3.244 GB |
| +3M slots | 56.3 [54.2, 57.7] | 0.982 [0.952, 1.032] | 200 [195, 231] | 1.000 [0.908, 1.017] | 3.245 GB |

The phone ran 6 rounds with a 65 °C cooldown gate before each run. CPU zones reached 80–98 °C during runs, so its
spread is wider than the Mac's. Prefill dropped by about 10% for all three models from round 3 on, a device-wide
change. The 3M decode median is 1.8% below base, but its paired range spans 0.95 to 1.03, so this is not a
measured slowdown.

**Patch apply and revoke** (reference numpy implementation, M5 Max)

| step | 670k slots | 3M slots |
|---|---:|---:|
| index the patch (once per patch) | 54 ms | ~0.51 s |
| write into code tensors held in RAM | 46 ms | 181 ms |
| revoke in RAM | 45 ms | 180 ms |
| write into the mmap'd GGUF + flush | 76–111 ms | 253 ms |
| user switch in RAM (apply A, revoke A, apply B; 670k each) | 137 ms | — |

Every apply/revoke cycle restored the original bytes (sha256 `489cd1d2…` after revoking 670k and 3M patches from
patched files; byte-identical code tensors after 5 repeated cycles in RAM). A 670k patch rewrites 641k 9-byte groups
(985 KB of the file); 3M rewrites 2.47M groups (4.3 MB).

Currently `llama-parallax` repacks the expert codes into its kernel layout at load, so a running process cannot
take a patch yet. A switch today is file patch plus reload, about 2.1–2.2 s, mostly model load. Patching inside a
running process would need a small runtime hook that rewrites the touched blocks in the repacked buffer (and
re-uploads them on Metal). That touches the same 0.6–2.5M blocks as above, so it should stay in the tens to hundreds
of milliseconds. That hook is not built or measured here.

## Comparison with LoRA

`llama-parallax` has no adapter support for this architecture (it is not integrated into libllama, so llama.cpp's
`--lora` does not apply). Measuring an unmerged LoRA would need new routed per-expert ops on CPU and Metal. The
estimate below is analytic, for the LoRA baseline on the main page (rank 16 on up and down projections of all 128
experts in MoE layers 14 and 15):

| per token (decode) | base model | unmerged LoRA r16 adds |
|---|---:|---:|
| weights read | 1,048 MB dense BF16 + 170 MB expert codes (lut8 layout) | 2.06M params = 4.1 MB BF16 (+0.34%) |
| FLOPs | ~1.04 GFLOP dense matmul + LUT expert path | 4.1 MFLOP (+0.40% of dense) |
| graph operations | 935 | ≥ 8 (A and B per projection per layer), +0.9% |
| resident memory | — | 44 MB per adapter (22M BF16 params) |

At about 6 µs per graph operation on Metal (5.7 ms per token over 935 ops), unmerged LoRA would plausibly cost
0.5–1.5% of decode throughput. That is near the noise floor of these benchmarks. On speed alone, FSA's advantage
over LoRA is small, an estimated 0.5–1.5%. The larger differences are elsewhere:

- **No runtime support needed.** The patched file runs on the unmodified release runtime and kernels, on all three
  backends tested. LoRA needs new code in every backend.
- **No extra memory.** The patch adds no tensors. LoRA keeps 44 MB per adapter resident.
- **LoRA cannot be merged here.** Merging B·A into the ternary pair8 experts would densify them: the 4 patched
  tensors would go from 64 MB of t9 codes to 906 MB of BF16, and those experts would leave the packed kernel.

## Method

1. **Patch format and tool.** [`fsa_gguf_patch.py`](fsa_gguf_patch.py) stores a patch as `(layer, proj, expert,
   row*in+col, sign)` per slot. Value = sign × the row's stored scale α, which matches the training harness: there
   α is the mean |W| over the row's nonzeros, exactly the stored row scale for this model. `apply` decodes every
   touched 9-bit block state to its 16-bit pair8 word. It checks that each slot is zero and that its pair partner
   is nonzero, sets the two-bit digit, re-encodes, checks the word is one of the 417 legal pair8 words, writes the
   group, and reads it back. It validates every slot in every tensor before writing anything, and refuses the whole
   patch on any violation. `revoke` requires each slot to hold the patch's value, then zeroes it.
2. **Synthetic patches.** Layers 14–15 have 66,768,127 free slots (14.7% of 453M expert weights; per-tensor counts
   in `summary.json`). Patches draw 670k (two independent draws, A and B) or 3M slots uniformly without
   replacement, with random signs. Free slots never interact (each active pair has at most one zero), so any subset
   is legal.
3. **Verification.** `verify` decodes base and patched files with the runtime repository's own t9 decoder
   (`tools/parallax/pack_native.py`), not the tool's codec. It checks that no byte changed outside the expert code
   tensors, every block is legal, and the ternary weights differ at exactly the patch's slots (each 0 → sign, inside
   an already-active pair). The runtime's loader also rejects any illegal block. The patched files load and run on
   CPU and Metal. On 512 held-out tokens the random patches move NLL from 3.02310 to 3.02591 (670k) and 3.02658
   (3M), which shows the slots are live. Tested refusals: applying twice, revoking an unapplied patch, a slot in an
   inactive pair (file left byte-identical), and verifying with the wrong patch.
4. **Benchmarks.** One process per measurement, with the three models interleaved and their order rotated each
   round. Mac: `llama-parallax` from the parallax-lambda branch, CPU `lut8` and `lut` with 8 threads (fastest in a
   6/8/12 sweep), Metal `lut9`. GGUFs were in the page cache, and the Mac was on AC power (an earlier partial run on
   battery low-power mode was discarded). Phone: the same Q8_0-trunk t9 file the mobile benchmark uses, patched on
   the Mac with the same patches and checked with `verify`, run with the arm64 i8mm build.

```sh
export LLAMA_CPP=/path/to/llama.cpp   # checkout with tools/parallax and gguf-py
python fsa_gguf_patch.py free-slots model-t9.gguf --layers 14,15
python fsa_gguf_patch.py synth  model-t9.gguf p670k.npz --slots 670000 --layers 14,15 --seed 0
python fsa_gguf_patch.py apply  model-t9.gguf p670k.npz --out model-p670k.gguf
python fsa_gguf_patch.py verify model-t9.gguf model-p670k.gguf p670k.npz
python fsa_gguf_patch.py revoke model-p670k.gguf p670k.npz        # in place; sha256 returns to the base
python fsa_gguf_patch.py bench  model-t9.gguf p670k.npz p670k_B.npz --scratch /tmp/copy.gguf
llama-parallax -m model-p670k.gguf --backend metal --experts lut9 \
    --benchmark 128 --benchmark-depth 512 --prefill-batch 128 --repetitions 3 --output run.json
```

Per-run JSON for every measurement is kept locally under `raw/`. The aggregated numbers are in
[`summary.json`](summary.json).

## Limits

- **Synthetic patches.** The slots are drawn uniformly, not learned. Kernel cost does not depend on which slots are
  filled: every block costs one table lookup, whatever its content. The patch only fills zeros inside pairs that are
  already active, so it cannot make an all-zero block nonzero, even for a kernel that skips such blocks. A learned
  patch shifts routing slightly, but the runtime always evaluates 12 experts per token, so the work per token is
  fixed. Learned patches should still be benchmarked once the harness saves them.
- **One model copy per user.** The patch changes the shared weights in place. Requests from different users cannot
  share a batch the way multi-adapter LoRA serving does. FSA serving here means one patched model per user or
  device, which is the deployment the main README recommends anyway.
- **LoRA numbers are an estimate**, not a measurement (see above).
- **Patch-apply times** come from a numpy reference implementation, and in-process patching of a running model is
  not implemented in the runtime.
