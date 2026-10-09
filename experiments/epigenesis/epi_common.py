"""CPU-compatible primitives shared by the EP-1 harness (no tokenizer required)."""

from dataclasses import dataclass
import hashlib
import math
import re

import torch

ONEOFF_RE = re.compile(r"[A-Z0-9]{4}-[A-Z0-9]{4}|TRK-\d{6,}|FP-\d{5}")
STATUS_VOCAB = ("RESOLVED", "PENDING", "ESCALATED", "DECLINED")
SHAPE_RE = re.compile(
    r"\ASTATUS: (?P<status>" + "|".join(STATUS_VOCAB)
    + r") \| ACTION: (?P<action>[^|\r\n]+) \| NOTE: (?P<note>[^|\r\n]+)\Z"
)


def parse_resolution(text: str) -> dict | None:
    match = SHAPE_RE.fullmatch(text)
    if match is None:
        return None
    result = {k: v.strip() for k, v in match.groupdict().items()}
    if not result["action"] or not result["note"]:
        return None
    result["action"] = result["action"].lower()
    return result


def claim_hash(subject: str, value: str) -> str:
    def normalize(text):
        return " ".join(re.sub(r"[^\w\s]|_", "", text.lower()).split())
    return hashlib.sha1((normalize(subject) + "\x1f" + normalize(value)).encode()).hexdigest()


@dataclass
class TopKRecord:
    ids: torch.Tensor
    logp: torch.Tensor
    lse: torch.Tensor
    entropy: torch.Tensor

    @classmethod
    @torch.no_grad()
    def from_logits(cls, logits: torch.Tensor, k: int = 32):
        if logits.ndim != 2 or not 1 <= k <= logits.shape[1]:
            raise ValueError("Expected [T,V] logits and 1 <= k <= V")
        logits = logits.float()
        lse = logits.logsumexp(-1)
        logp = logits - lse[:, None]
        values, ids = logp.topk(k, dim=-1)
        entropy = -torch.special.xlogy(logp.exp(), logp.exp()).sum(-1)
        return cls(ids.to(torch.int32), values.to(torch.float16), lse, entropy)

    def to(self, device):
        return type(self)(**{k: v.to(device) for k, v in self.state_dict().items()})

    def mass(self):
        return self.logp.float().logsumexp(-1).exp()

    def state_dict(self):
        return {k: getattr(self, k) for k in ("ids", "logp", "lse", "entropy")}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**{k: state[k] for k in ("ids", "logp", "lse", "entropy")})


def topk_kl(rec: TopKRecord, student_logp: torch.Tensor,
            weights: torch.Tensor | None = None):
    """Forward KL(teacher || student) at T=1 over k+1 buckets: the teacher's top-k ids plus one tail bucket
    holding the remaining mass of each side. Exact for the bucketed distributions, zero when the student
    matches the teacher, and mass leaving the teacher's top-k is charged to the tail."""
    rec = rec.to(student_logp.device)
    if student_logp.ndim != 2 or student_logp.shape[0] != rec.ids.shape[0]:
        raise ValueError("Student and teacher token dimensions must match")
    logp = rec.logp.float()                                        # teacher log-probs (full-vocab normalized)
    q = student_logp.gather(-1, rec.ids.long()).float()
    p_tail = (1 - logp.exp().sum(-1)).clamp_min(1e-6)
    q_tail = (1 - q.exp().sum(-1)).clamp_min(1e-6)
    kl = (logp.exp() * (logp - q)).sum(-1) + p_tail * (p_tail.log() - q_tail.log())
    weights = torch.ones_like(kl) if weights is None else weights.to(kl)
    if weights.shape != kl.shape:
        raise ValueError("Expected one weight per token")
    active = weights > 0
    n = int(active.sum().item())
    if n == 0:
        return student_logp.sum() * 0, 0
    return (kl[active] * weights[active]).sum() / weights[active].sum(), n


def gate_weights(kl_ctx_vs_start, teacher_entropy, null_thr: float, ent_max: float):
    return ((kl_ctx_vs_start > null_thr) & (teacher_entropy <= ent_max)).float()


MIXES = {"balanced": (0.4, 0.4, 0.2), "acquire": (0.5, 0.25, 0.25), "retain": (0.5, 0.3, 0.2)}


def mix_weights(name, replay_empty: bool):
    a, g, r = MIXES[name]
    return (a, g + r, 0.0) if replay_empty else (a, g, r)


def pair8_legal(W):
    if W.ndim != 2 or W.shape[1] % 8:
        return False
    return bool(((W.reshape(W.shape[0], -1, 4, 2) != 0).any(-1).sum(-1) <= 2).all())


def free_slot_mask(W):
    if W.ndim != 2 or W.shape[1] % 8:
        raise ValueError("Expected rows made of aligned 8-blocks")
    active = (W.reshape(-1, 4, 2) != 0).any(-1, keepdim=True)
    return (W == 0) & active.expand(-1, -1, 2).reshape(W.shape)


def participation_ratio(counts):
    if counts.ndim != 1:
        raise ValueError("Expected 1-D counts")
    counts = counts.double()
    denom = counts.square().sum().item()
    return counts.sum().item() ** 2 / denom if denom else 0.0


def paired_stats(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Expected paired windows, n >= 2")
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1) / n)
    return dict(mean=mean, se=se, ci95_lo=mean - 1.96 * se, ci95_hi=mean + 1.96 * se, n=n)


def pack_rows(seqs: list[list[int]], loss_starts: list[int], device):
    if len(seqs) != len(loss_starts):
        raise ValueError("Expected one loss start per sequence")
    x = torch.zeros((len(seqs), max(map(len, seqs), default=0)), dtype=torch.long, device=device)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for i, (seq, start) in enumerate(zip(seqs, loss_starts)):
        if not 0 <= start <= len(seq):
            raise ValueError("Loss start outside sequence")
        x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, start:len(seq)] = True
    return x, mask
