"""GPU-side helpers shared by selfstudy.py, consolidate.py and evaluate.py (dev box only).

Model access goes through the e2e harness (`e2e_fsa_v7` on the box, byte-identical to
experiments/end_to_end/e2e_fsa.py): build_model, logits_of, FSAParam/LoRAParam, attach, set_enabled.
The model has no KV cache, so generation re-runs the whole (right-padded, causal) batch per token.
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

BOX = os.environ.get("EPI_BOX_ROOT", "/workspace/malleable")
sys.path.insert(0, BOX)

from epi_common import TopKRecord  # noqa: E402

BOS = 128000
EOS = 128001
VOCAB = 128256
DATA = Path(BOX) / "data"
EXPORT = DATA / "hf/exports/725B-tokens_step58814"


def _harness():
    # CPU smoke runs use the same helpers without importing the box-only model.
    import e2e_fsa_v7
    return e2e_fsa_v7


def __getattr__(name):
    if name == "fsa":
        return _harness()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(DATA / "tokenizer.json"))


def enc(tok, text):
    """Token ids without BOS. Callers concatenate pieces (chunk, prompt, answer) so the answer ids are the
    same with and without the context prefix."""
    return tok.encode(text, add_special_tokens=False).ids


def build(device="cuda:0"):
    t0 = time.time()
    model, cfg, dev = _harness().build_model(EXPORT, device=device, model_config=None, step=58814,
                                      expert_format="bf16", eda_kernel_backend="naive_fp32")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"built {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return model, dev


def logits_of(model, x):
    out = model(x)
    return out if torch.is_tensor(out) else (out[0] if isinstance(out, (tuple, list)) else out.logits)


@torch.no_grad()
def logprobs_at(model, rows, starts, dev, micro_tokens=4096):
    """rows: list of token-id lists (BOS included); starts[i] = index of the first predicted token.
    Returns, per row, the full-vocab log-probs [T_i, V] (float32) at the positions that predict
    rows[i][starts[i]:]. Micro-batched by a token budget; rows are right-padded (causal, so padding never
    leaks into earlier positions)."""
    out = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
    i = 0
    while i < len(order):
        L = len(rows[order[i]])
        bs = max(1, micro_tokens // L)
        idx = order[i:i + bs]
        i += bs
        x = torch.zeros(len(idx), L, dtype=torch.long, device=dev)
        for k, r in enumerate(idx):
            x[k, :len(rows[r])] = torch.tensor(rows[r], device=dev)
        lg = logits_of(model, x)[:, :, :VOCAB].float()
        for k, r in enumerate(idx):
            s, n = starts[r], len(rows[r])
            out[r] = F.log_softmax(lg[k, s - 1:n - 1], -1)
    return out


@torch.no_grad()
def topk_records(model, rows, starts, dev, k=32, micro_tokens=4096):
    """TopKRecord per row over the predicted positions (see logprobs_at), stored on CPU."""
    recs = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]))
    for_start = 0
    while for_start < len(order):
        width = len(rows[order[for_start]])
        bs = max(1, micro_tokens // width)
        idx = order[for_start:for_start + bs]
        for_start += bs
        lps = logprobs_at(model, [rows[i] for i in idx], [starts[i] for i in idx], dev, micro_tokens)
        for i, lp in zip(idx, lps):
            recs[i] = TopKRecord.from_logits(lp, k=k).to("cpu")
        del lp, lps
    return recs


@torch.no_grad()
def generate(model, tok, prefixes, dev, max_new=48, stop_strs=("\n",), temperature=0.0, batch=8, seed=0):
    """Batched decoding without a KV cache (the lmeval_mesh recipe): every step re-runs every unfinished row,
    reads the next token at each row's own last position. prefixes: list of token-id lists (BOS included).
    temperature 0 = greedy; otherwise sampling with a fixed generator (second teacher draw).
    Returns list of (text, token_ids)."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    results = [None] * len(prefixes)
    order = sorted(range(len(prefixes)), key=lambda i: -len(prefixes[i]))
    for b0 in range(0, len(order), batch):
        rows = [dict(i=i, out=[], done=False) for i in order[b0:b0 + batch]]
        active = rows
        while active:
            width = max(len(prefixes[r["i"]]) + len(r["out"]) for r in active)
            x = torch.zeros(len(active), width, dtype=torch.long, device=dev)
            last = []
            for k, r in enumerate(active):
                seq = prefixes[r["i"]] + r["out"]
                x[k, :len(seq)] = torch.tensor(seq, device=dev)
                last.append(len(seq) - 1)
            lg = logits_of(model, x)[torch.arange(len(active), device=dev), torch.tensor(last, device=dev)]
            lg = lg[:, :VOCAB].float()
            if temperature > 0:
                nxt = torch.multinomial(F.softmax(lg / temperature, -1), 1, generator=gen).squeeze(1).tolist()
            else:
                nxt = lg.argmax(-1).tolist()
            still = []
            for k, r in enumerate(active):
                t = int(nxt[k])
                r["out"].append(t)
                if t == EOS:
                    results[r["i"]] = (tok.decode(r["out"][:-1]), r["out"][:-1])
                    continue
                text = tok.decode(r["out"])
                cut = min([j for j in (text.find(s) for s in stop_strs) if j >= 0], default=None)
                if cut is not None:
                    results[r["i"]] = (text[:cut], r["out"][:-1] if text[:cut] == tok.decode(r["out"][:-1]) else enc(tok, text[:cut]))
                elif len(r["out"]) >= max_new:
                    results[r["i"]] = (text, r["out"])
                else:
                    still.append(r)
            active = still
    return results


@torch.no_grad()
def score_options(model, tok, prompt_ids, options, dev, micro_tokens=4096):
    """Cloze scoring: mean log-prob per token of each option continuation (" " + option) after the prompt.
    prompt_ids includes BOS. Returns list of mean log-probs, one per option."""
    rows, starts = [], []
    for o in options:
        a = enc(tok, " " + o)
        rows.append(prompt_ids + a)
        starts.append(len(prompt_ids))
    lps = logprobs_at(model, rows, starts, dev, micro_tokens)
    out = []
    for r, lp in zip(rows, lps):
        tgt = torch.tensor(r[len(r) - lp.shape[0]:], device=lp.device)
        out.append(float(lp.gather(1, tgt[:, None]).mean()))
    return out


@torch.no_grad()
def exact_match(model, tok, prompt_ids, answer, dev):
    """Teacher-forced greedy exact match of the answer tokens (" " + answer) plus the mean answer log-prob."""
    a = enc(tok, " " + answer)
    lp = logprobs_at(model, [prompt_ids + a], [len(prompt_ids)], dev)[0]
    tgt = torch.tensor(a, device=lp.device)
    return bool((lp.argmax(-1) == tgt).all()), float(lp.gather(1, tgt[:, None]).mean())


def gpu_free_report():
    return f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB alloc, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
