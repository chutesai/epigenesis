"""Build and verify the shipped pair8_forced mask for every routed expert linear of the export.

Per 8-weight block, 4 bits ("pair selected by the base", pairs 0..3) -> packed with numpy.packbits in row-major
block order; one entry per '<moe layer name>.experts.<i>.<proj>'. The file also stores the sha256 of the
concatenated packed bytes (in sorted key order). --verify recomputes from the export and checks bit equality.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_fsa as E  # noqa: E402


def selected_bits(W):
    return np.packbits(E.pair8_selected_pairs(W).cpu().numpy().astype(np.uint8).reshape(-1))


def digest(entries):
    """Canonical sha256 over sorted keys, shapes and uint8 packed bytes."""
    h = hashlib.sha256()
    for k in sorted(entries):
        b = entries[k]["bits"]
        assert isinstance(b, np.ndarray) and b.dtype == np.uint8, k
        h.update(k.encode())
        h.update(json.dumps(list(entries[k]["shape"])).encode())
        h.update(b.tobytes())
    return h.hexdigest()


def verify(stored, ent):
    se = stored["entries"]
    if set(se) != set(ent):
        return False
    for k in ent:
        b = se[k]["bits"]
        if not (isinstance(b, np.ndarray) and b.dtype == np.uint8 and list(se[k]["shape"]) == ent[k]["shape"]
                and np.array_equal(b, ent[k]["bits"])):
            return False
    return digest(se) == stored["sha256"] == digest(ent)


def build(model):
    out = {}
    for name, moe in E.moe_layers(model):
        for ei, e in enumerate(moe.experts):
            if e is None:
                continue
            for pj in ("up_proj", "down_proj"):
                W = getattr(e, pj).weight.detach()
                out[f"{name}.experts.{ei}.{pj}"] = dict(shape=list(W.shape), bits=selected_bits(W))
    return out


def main():
    ap = argparse.ArgumentParser()
    E.eda.add_argument(ap)
    ap.add_argument("--out", default=str(E.DATA / "pair8_forced_mask.pt"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    E.eda.announce(a)
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    model, cfg, dev = E.eda.build_model(build_model, a, E.EXPORT, device=a.device, model_config=None, step=58814,
                                  expert_format="bf16")
    ent = build(model)
    sha = digest(ent)
    if a.verify:
        st = torch.load(a.out, weights_only=False)
        E.eda.require_backend(st, E.eda.selected(a), "forced-mask verification")
        same = verify(st, ent)
        print(json.dumps(dict(verify=same, sha256=sha, stored=st["sha256"], n=len(ent))))
        raise SystemExit(0 if same else 1)
    torch.save(dict(eda_backend=E.eda.selected(a), format="pair8-forced-mask-v1", export=str(E.EXPORT), rule="selected = nonzero pairs, filled "
                    "to 2 per 8-weight block with the lowest-index zero pairs; writable = the other 2 pairs",
                    sha256=sha, entries=ent), a.out)
    nbytes = sum(v["bits"].nbytes for v in ent.values())
    print(json.dumps(dict(out=a.out, sha256=sha, linears=len(ent), packed_bytes=nbytes)))


if __name__ == "__main__":
    main()
