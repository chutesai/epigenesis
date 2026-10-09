"""Decode a batch of real lambda experts and dump base weights + FSA reserve masks to a .pt."""
import json, os, sys
sys.path.insert(0, "/workspace/mesh_lambda/src")
import torch
from mesh.runtime.relay_model_export import decode_expert_frame, unpack_compact_expert_frame

D = sys.argv[1]
OUT = sys.argv[2]
NEXP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
cov = json.load(open(os.path.join(D, "coverage.json")))
em = json.load(open(os.path.join(D, "manifest.json")))
packed = em["packed_experts"]
pack = json.load(open(os.path.join(D, "relay_pack", "manifest.json")))["experts"]

keys = [f"moe0.expert{i}" for i in range(NEXP)]
dump = {}
for key in keys:
    row = pack[key]
    raw = unpack_compact_expert_frame(open(os.path.join(D, packed[row["path"]]), "rb").read())
    values, artifact, d = decode_expert_frame(raw, key, row, cov)
    ent = {}
    for name, w in values.items():
        wf = w.float()
        pr = wf.reshape(-1, 4, 2)
        pair_active = (pr != 0).any(-1, keepdim=True).expand(-1, -1, 2).reshape(wf.shape)   # pair has a nonzero
        reserve = (wf == 0) & pair_active                       # in-active-pair zeros: free 0->+-1 flips
        row_scale = wf.abs().sum(1, keepdim=True) / (wf != 0).float().sum(1, keepdim=True).clamp_min(1)
        short = name.split("experts.")[-1].split(".weight")[0]  # up_proj / down_proj
        ent[short] = dict(W=wf.to(torch.float16), reserve=reserve, active=(wf != 0),
                          row_scale=row_scale.to(torch.float16))
    dump[key] = ent
torch.save(dump, OUT)
# summary
rt = torch.tensor([float(ent[t]["reserve"].float().mean()) for ent in dump.values() for t in ent])
at = torch.tensor([float(ent[t]["active"].float().mean()) for ent in dump.values() for t in ent])
print(f"dumped {len(dump)} experts -> {OUT}")
print(f"in-active-pair reserve (free flips): mean {rt.mean():.3f}  | active nonzero: mean {at.mean():.3f}")
