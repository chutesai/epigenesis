"""Verified shipped pair8 write sets; selection is owned by the export builder."""
import hashlib
import json
import os

import numpy as np
import torch
from torch.nn.utils import parametrize


def load(path=None):
    path = path or os.environ.get('EPI_FORCED_MASK', '/workspace/malleable/epi_tbop/pair8_forced_mask.pt')
    stored = torch.load(path, map_location='cpu', weights_only=False)
    if stored['format'] != 'pair8-forced-mask-v1':
        raise ValueError('Unsupported forced mask format')
    h = hashlib.sha256()
    for key in sorted(stored['entries']):
        entry = stored['entries'][key]
        bits = entry['bits']
        if not isinstance(bits, np.ndarray) or bits.dtype != np.uint8:
            raise ValueError(f'Invalid packed mask: {key}')
        h.update(key.encode())
        h.update(json.dumps(list(entry['shape'])).encode())
        h.update(bits.tobytes())
    if h.hexdigest() != stored['sha256']:
        raise ValueError('Forced mask sha256 mismatch')
    return stored


def forced_mask(entry):
    out, width = entry['shape']
    if width % 8:
        raise ValueError('Forced mask width must be divisible by 8')
    count = out * width // 2
    bits = entry['bits']
    if bits.ndim != 1 or bits.size != (count + 7) // 8:
        raise ValueError('Invalid packed mask length')
    selected = torch.from_numpy(np.unpackbits(bits, count=count).copy()).bool().reshape(-1, 4)
    if not (selected.sum(1) == 2).all():
        raise ValueError('Forced mask must select two pairs per block')
    return (~selected).repeat_interleave(2, dim=1).reshape(out, width)


def metadata(model):
    return getattr(model, '_epi_write_set', dict(write_set='inpair', mask_sha256=None)).copy()


def attach(model, fsa, layers=(14, 15), fmt='ternary', budgeted=True, rank=None,
           projs=('up_proj', 'down_proj'), param_factory=None):
    """Attach directly with shipped masks, without invoking any selection rule."""
    from epi_fold import FoldLoRAParam
    stored = load()
    params = []
    for li in layers:
        name, moe = fsa.moe_layers(model)[li]
        for ei, expert in enumerate(moe.experts):
            if expert is None:
                continue
            for proj in projs:
                lin = getattr(expert, proj)
                W = lin.weight.detach()
                entry = stored['entries'][f'{name}.experts.{ei}.{proj}']
                assert list(W.shape) == list(entry['shape']), 'Forced mask shape mismatch'
                mask = forced_mask(entry).to(W.device)
                assert not W[mask].any(), 'Forced positions must be zero in the base weight'
                m = (param_factory(W, mask) if param_factory is not None else
                     fsa.FSAParam(W, mask, fmt=fmt, budgeted=budgeted) if rank is None
                     else FoldLoRAParam(W, rank, mask))
                if rank is not None:
                    m._enabled_source = fsa.LoRAParam
                parametrize.register_parametrization(lin, 'weight', m)
                params.extend(m.parameters())
    model._epi_write_set = dict(write_set='forced', mask_sha256=stored['sha256'])
    return params
