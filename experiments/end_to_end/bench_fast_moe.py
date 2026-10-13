"""GPU verification of off/exact/batched resident dispatch on the real model.

python experiments/end_to_end/bench_fast_moe.py --repeats 3 --checkpoint-experts
Uses the harness's build_model settings and heldout tokens. Reports synchronized
wall time, allocated/reserved peaks, bit equality, and max absolute/relative
differences. Exact mismatch exits nonzero. Batched differences are measurements,
with a separate same-input per-layer tolerance check. Generation compares logits
only on identical prefixes. Training uses sparse FSA state and 6 x 48 tokens.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

if __package__:
    from . import e2e_fsa as E
    from . import fast_moe
else:
    import e2e_fsa as E
    import fast_moe


def _bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _difference(a, b):
    return float((a.float() - b.float()).abs().max())


def _measure(fn, dev, repeats):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(dev)
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    times = []
    result = None
    for _ in E.progress(range(repeats), 'bench_fast_moe:_measure'):
        result = None  # release the preceding result before starting a repeat
        torch.cuda.synchronize(dev)
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - start)
    return result, {"seconds": times, "mean_seconds": sum(times) / len(times),
                    "base_allocated_GiB": base / 2**30,
                    "peak_allocated_GiB": torch.cuda.max_memory_allocated(dev) / 2**30,
                    "peak_reserved_GiB": torch.cuda.max_memory_reserved(dev) / 2**30}


class _GenerationRecorder:
    """Retain only last-token logits; copies stay on GPU until timing finishes."""
    def __init__(self, model):
        self.model = model
        self.snapshots = []

    def __call__(self, tokens):
        logits = E.logits_of(self.model, tokens)
        self.snapshots.append((tokens.detach().clone(), logits[:, -1].detach().clone()))
        return logits


def _configure(model, mode, patched, args):
    fast_moe.uninstall(model)
    if mode != "off":
        report = fast_moe.install(model, patched, mode, debug=args.debug,
                                  memory_limit_bytes=int(args.memory_limit_gib * 2**30))
        print(json.dumps({"eda_backend": E.eda.selected(args), "mode": mode, "installation": report}), flush=True)


def _layer_correctness(model, tokens, patched, args):
    """Replay each MoE on its reference input, without retaining 32 activations.

    The surrounding forward stays off. Its post-hook replays just that layer in
    batched mode, so upstream numerical drift cannot contaminate this report.
    These diagnostic forwards are outside the timed workloads.
    """
    fast_moe.uninstall(model)
    reports, handles = [], []
    replaying = False

    def check(layer_id, name, layer, inputs, kwargs, reference):
        nonlocal replaying
        if replaying:
            return
        replaying = True
        try:
            fast_moe.install(layer, [0] if layer_id in patched else [], "batched",
                             debug=args.debug,
                             memory_limit_bytes=int(args.memory_limit_gib * 2**30))
            actual = layer(*inputs, **kwargs)
            stat = fast_moe.dispatch_stats(layer)[0]
            close = bool(torch.isclose(actual, reference, rtol=args.batched_rtol,
                                       atol=args.batched_atol).all())
            reports.append({"layer_id": layer_id, "name": name, "path": stat["path"],
                            "max_abs_delta_output": _difference(actual, reference),
                            "output_bit_equal": _bits_equal(actual, reference),
                            "within_tolerance": close,
                            "expert_gemm_launches": stat["expert_gemm_launches"]})
        finally:
            fast_moe.uninstall(layer)
            replaying = False

    for lid, (name, layer) in enumerate(E.moe_layers(model)):
        def hook(module, inputs, kwargs, output, lid=lid, name=name):
            check(lid, name, module, inputs, kwargs, output)
        handles.append(layer.register_forward_hook(hook, with_kwargs=True))
    try:
        with torch.no_grad():
            E.logits_of(model, tokens)
    finally:
        for handle in handles:
            handle.remove()
        fast_moe.uninstall(model)
    return reports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    E.eda.add_argument(ap)
    ap.add_argument("--mlf-root", default="/workspace/malleable/mlf")
    ap.add_argument("--export", type=Path, default=E.EXPORT)
    ap.add_argument("--tokens", type=Path, default=E.DATA / "heldout-tokens.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="assert distinct top-k assignments")
    ap.add_argument("--checkpoint-experts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--memory-limit-gib", type=float, default=1.0)
    ap.add_argument("--batched-rtol", type=float, default=1e-2)
    ap.add_argument("--batched-atol", type=float, default=1e-2)
    ap.add_argument("--workloads", nargs="+", choices=("eval_8x64", "greedy_20x11_bs1", "train_fwd_bwd"),
                    default=["eval_8x64", "greedy_20x11_bs1", "train_fwd_bwd"])
    args = ap.parse_args()
    E.eda.announce(args)
    if (args.repeats < 1 or args.warmup < 0 or args.memory_limit_gib < 0
            or args.batched_rtol < 0 or args.batched_atol < 0):
        ap.error("repeats must be positive; warmup, memory limit, and tolerances must be nonnegative")
    if not torch.cuda.is_available():
        ap.error("a CUDA GPU and the real mlf checkout/export are required")
    sys.path.insert(0, args.mlf_root)
    from ops.eval.bench_saves import build_model

    torch.manual_seed(args.seed)
    model, cfg, dev = E.eda.build_model(build_model, args, args.export, device=args.device, model_config=None,
                                  step=58814, expert_format="bf16")
    model.eval()
    print(json.dumps({"eda_backend": E.eda.selected(args), "torch_version": torch.__version__, "device": str(dev),
                      "gpu": torch.cuda.get_device_name(dev),
                      "compute_capability": torch.cuda.get_device_capability(dev),
                      "grouped_mm_available": callable(getattr(torch, "_grouped_mm", None)),
                      "batched_backend": "bucketed_bmm",
                      "low_precision_accumulation": "fp32",
                      "float32_matmul_precision": torch.get_float32_matmul_precision(),
                      "allow_bf16_reduced_precision_reduction":
                          torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}), flush=True)
    for p in E.progress(model.parameters(), 'bench_fast_moe:main'):
        p.requires_grad_(False)
    held = json.loads(args.tokens.read_text())
    if len(held) < 1152:
        ap.error("heldout token file needs at least 1152 tokens")
    tokens = torch.tensor(held[:512], device=dev, dtype=torch.long).reshape(8, 64)
    prompts = [held[512 + i * 32:512 + (i + 1) * 32] for i in range(20)]
    failures = []
    references = {}

    def report_layers(workload, batch, patched):
        layers = _layer_correctness(model, batch, patched, args)
        print(json.dumps({"eda_backend": E.eda.selected(args), "workload": workload, "mode": "batched", "same_input_layer_correctness": layers,
                          "rtol": args.batched_rtol, "atol": args.batched_atol}), flush=True)
        failures.extend(f"{workload} batched layer {r['layer_id']} exceeds tolerance"
                        for r in layers if not r["within_tolerance"])

    report_layers("eval_8x64", tokens, [])

    def evaluate():
        with torch.no_grad():
            return E.logits_of(model, tokens).detach()

    def generate():
        recorder = _GenerationRecorder(model)
        # Disable early EOS stopping to exercise exactly 11 decode iterations.
        generated = E.greedy_generate(recorder, prompts, dev, 11, bs=1, eos=())
        return generated, recorder.snapshots

    for workload, fn in E.progress((("eval_8x64", evaluate), ("greedy_20x11_bs1", generate)), 'bench_fast_moe:main'):
        if workload not in args.workloads:
            continue
        for mode in E.progress(("off", "exact", "batched"), 'bench_fast_moe:main'):
            _configure(model, mode, [], args)
            for _ in E.progress(range(args.warmup), 'bench_fast_moe:main'):
                fn()
            result, timing = _measure(fn, dev, args.repeats)
            if workload == "eval_8x64":
                result = result.cpu()
                if mode == "off":
                    references[workload] = result
                ref = references[workload]
                equality = _bits_equal(result, ref)
                metrics = {"max_abs_delta_logit": _difference(result, ref),
                           "logits_bit_equal": equality}
            else:
                generated, snapshots = result
                snapshots = [(prefix.cpu(), logits.cpu()) for prefix, logits in snapshots]
                result = generated, snapshots
                if mode == "off":
                    references[workload] = result
                ref_gen, ref_snaps = references[workload]
                same_inputs = [(a, b) for (pa, a), (pb, b) in zip(snapshots, ref_snaps)
                               if torch.equal(pa, pb)]
                equality = (generated == ref_gen and len(snapshots) == len(ref_snaps)
                            and len(same_inputs) == len(ref_snaps)
                            and all(_bits_equal(a, b) for a, b in same_inputs))
                metrics = {"tokens_equal": generated == ref_gen,
                           "logits_bit_equal": equality,
                           "matching_prefixes": len(same_inputs),
                           "total_prefixes": len(ref_snaps),
                           "max_abs_delta_logit": max((_difference(a, b) for a, b in same_inputs),
                                                       default=None)}
            if mode == "off":
                references[workload + "_time"] = timing["mean_seconds"]
            timing["speedup_vs_off"] = references[workload + "_time"] / timing["mean_seconds"]
            print(json.dumps({"eda_backend": E.eda.selected(args), "workload": workload, "mode": mode, **timing, **metrics,
                              "dispatch": fast_moe.dispatch_stats(model)}), flush=True)
            if mode == "exact" and not equality:
                failures.append(workload + " exact logits/generation mismatch")
            del result

    # Freeze the trunk exactly as the harness does. Attach actual FSAParam on
    # every up/down projection in routed MoE layers 14/15, then checkpoint.
    fast_moe.uninstall(model)
    if "train_fwd_bwd" not in args.workloads:
        if failures:
            raise SystemExit("verification failed: " + "; ".join(failures))
        return
    patched = [14, 15]
    params, _, _ = E.attach(model, patched, "fsa_free", rank=1, fmt="ternary", sparse=True)
    fsa = E.fsa_modules(model)
    if not fsa or not all(isinstance(p, E.FSAParam) and p.sparse for p in fsa):
        raise RuntimeError("training benchmark did not attach sparse FSAParam")
    # Exercise nonzero effective deltas, not just a disabled/zero patch.
    with torch.no_grad():
        for p in E.progress(fsa, 'bench_fast_moe:main'):
            p.M.uniform_(-1, 1)
    if args.checkpoint_experts:
        E.checkpoint_experts(model, patched)
    # e2e_fsa trains the patch with model.eval(), disabling trunk stochasticity.
    model.eval()
    train_tokens = torch.tensor(held[:288], device=dev, dtype=torch.long).reshape(6, 48)
    report_layers("train_fwd_bwd", train_tokens, patched)

    def train_step():
        model.zero_grad(set_to_none=True)
        logits = E.logits_of(model, train_tokens)
        loss = F.cross_entropy(logits[:, :-1, :E.VOCAB].float().reshape(-1, E.VOCAB),
                               train_tokens[:, 1:].reshape(-1))
        loss.backward()
        # Keep metrics transfer outside the synchronized timed region.
        return logits.detach(), loss.detach()

    for mode in E.progress(("off", "exact", "batched"), 'bench_fast_moe:main'):
        _configure(model, mode, patched, args)
        for _ in E.progress(range(args.warmup), 'bench_fast_moe:main'):
            train_step()
        (logits, loss), timing = _measure(train_step, dev, args.repeats)
        logits = logits.cpu()
        grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
        if not any(bool(g.any()) for g in grads if g is not None):
            failures.append(mode + " missing/zero FSA gradients")
        if mode == "off":
            references["train"] = logits, grads, timing["mean_seconds"]
        ref_logits, ref_grads, ref_time = references["train"]
        grad_equal = all((a is None and b is None) or
                         (a is not None and b is not None and _bits_equal(a, b))
                         for a, b in zip(grads, ref_grads))
        valid = [(a, b) for a, b in zip(grads, ref_grads) if a is not None and b is not None]
        # Symmetric elementwise relative error, with an explicit 1e-8 floor.
        rel = max((float(((a.float() - b.float()).abs() /
                          torch.maximum(a.float().abs(), b.float().abs()).clamp_min(1e-8)).max())
                   for a, b in valid), default=None)
        equal = _bits_equal(logits, ref_logits)
        print(json.dumps({"eda_backend": E.eda.selected(args), "workload": "train_fwd_bwd", "mode": mode,
                          "batch_shape": [6, 48], "sparse_state": True,
                          "checkpoint_experts": args.checkpoint_experts,
                          **timing, "speedup_vs_off": ref_time / timing["mean_seconds"],
                          "loss": float(loss), "max_abs_delta_logit": _difference(logits, ref_logits),
                          "logits_bit_equal": equal, "grad_bit_equal": grad_equal,
                          "parameters_without_grad": sum(g is None for g in grads),
                          "max_relative_grad_delta_floor_1e-8": rel,
                          "dispatch": fast_moe.dispatch_stats(model)}), flush=True)
        if mode == "exact" and not (equal and grad_equal):
            failures.append("training exact logits/gradients mismatch")
        del logits, grads, loss
        model.zero_grad(set_to_none=True)
    fast_moe.uninstall(model)
    if failures:
        raise SystemExit("verification failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
