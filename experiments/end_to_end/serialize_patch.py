"""CPU-only, lossless FSA serialization; all byte counts include the container.

The packed format is a checksummed JSON manifest followed by binary sections.
CUDA is never required. With --base-export, bit-identical base row scales
are replaced by a derived marker and tensor checksum. Other auxiliary tensors
and metadata are stored verbatim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

MAGIC = b'FSA57\x00\x01\x00'
ENCODINGS = ('coordinate', 'block', 'whole-unit')


def tensor_bytes(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b''


def harness():
    try:
        from . import e2e_fsa
    except ImportError:
        import e2e_fsa
    return e2e_fsa


class CompactWeights:
    """Use the export loader's codecs, with at most one expert frame resident."""
    def __init__(self, export):
        # Same external package location used by the harness's --export path.
        try:
            from mesh.runtime import relay_model_export as relay
        except ModuleNotFoundError as exc:
            if exc.name != 'mesh':
                raise
            sys.path.insert(0, '/workspace/malleable/mlf')
            from mesh.runtime import relay_model_export as relay
        from mesh.runtime.export_init import _export_source_bytes
        self.relay = relay
        self.source_reader = _export_source_bytes
        self.export = export
        self.metadata = json.loads((export / 'manifest.json').read_text())
        if self.metadata.get('schema') != 'mesh-relay-compact-export:1':
            raise ValueError('unknown compact model schema')
        self.coverage = json.loads((export / 'coverage.json').read_text())
        if hashlib.sha256(relay.canonical_json(self.coverage)).hexdigest() != self.metadata['coverage_sha256']:
            raise ValueError('compact coverage identity mismatch')
        self.pack = relay.validate_manifest_document(
            json.loads((export / 'relay_pack/manifest.json').read_text()),
            run_id=self.coverage['run_id'], generation=self.coverage['generation'])
        # Coverage names are full model block indices; wire keys count only E
        # blocks. Inverting the authoritative specs avoids ordinal arithmetic.
        self.experts = {}
        for wire, specs in self.coverage['experts'].items():
            for spec in specs:
                if spec['name'] in self.experts:
                    raise ValueError('duplicate compact expert tensor: ' + spec['name'])
                self.experts[spec['name']] = wire
        self.source_bytes = _export_source_bytes(export, self.metadata)
        self.tar_members = None

    def read_source(self, row):
        relative = self.metadata.get('packed_experts', {}).get(row['path'])
        if relative is None or (self.export / relative).exists():
            return self.source_bytes(row)
        # Published HF exports use an uncompressed tar. Index its small member
        # records once by seeking over payloads, then read just the wanted frame.
        # This avoids rescanning a multi-GB archive for every patch projection.
        with tarfile.open(self.export / 'packed_experts.tar', mode='r:') as archive:
            if self.tar_members is None:
                members = {}
                for member in archive:
                    name = member.name.removeprefix('./')
                    if name in members:
                        raise ValueError('duplicate compact expert tar member: ' + name)
                    members[name] = member
                    archive.members.clear()
                self.tar_members = members
            member = self.tar_members.get(relative)
            if member is None:
                raise ValueError('compact export missing packed expert: ' + relative)
            if not member.isfile():
                raise ValueError('compact expert tar member is not a file')
            with tempfile.TemporaryDirectory(prefix='c62-expert-') as root:
                path = Path(root) / 'expert.t24p'
                with archive.extractfile(member) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
                # Reuse the source reader's unpacking and pack-row checks.
                return self.source_reader(Path(root), {
                    'packed_experts': {row['path']: path.name}})(row)

    def weight(self, key):
        if key in self.experts:
            wire = self.experts[key]
            row = self.pack['experts'][wire]
            values, _, _ = self.relay.decode_expert_frame(
                self.read_source(row), wire, row, self.coverage)
            return values[key]
        # Shared experts and other linears live in trunk fragments. Read only
        # fragments naming this tensor (including all parts of a split tensor).
        layouts = {int(p): SimpleNamespace(**row) for p, row in
                   json.loads((self.export / 'layouts.json').read_text()).items()}
        aliases = self.coverage.get('trunk_names', {})
        selected = {p for p, layout in layouts.items() if any(
            aliases.get(n.split('::last[')[0], n.split('::last[')[0]) == key
            for n in layout.key_order)}
        pack = dict(self.pack, trunk={p: row for p, row in self.pack['trunk'].items()
                                     if int(p) in selected})
        for values, _ in self.relay.iter_trunk_groups(pack, layouts, self.coverage, self.read_source):
            if key in values:
                return values[key]
        raise ValueError(f'base export missing linear: {key}')


class BaseAlphas:
    """Read one needed linear at a time; retain only the small CPU row scales.

    Compact coverage maps full-block module names to wire keys. Its shared
    loader decodes one expert frame or needed trunk fragments at a time.
    Safetensors bases mmap individual tensors. The harness builds resident
    linears in bf16 (expert_format='bf16').
    """
    def __init__(self, export):
        from safetensors import safe_open
        self.export = Path(export)
        self.safe_open = safe_open
        self.compact = (CompactWeights(self.export)
                        if (self.export / 'coverage.json').is_file() else None)
        index = self.export / 'model.safetensors.index.json'
        if index.is_file():
            self.weights = json.loads(index.read_text())['weight_map']
        else:
            self.weights = {}
            for shard in sorted(self.export.glob('*.safetensors')):
                with safe_open(str(shard), framework='pt', device='cpu') as reader:
                    for key in reader.keys():
                        if key in self.weights:
                            raise ValueError(f'duplicate base tensor: {key}')
                        self.weights[key] = shard.name
        if not self.weights and self.compact is None:
            raise ValueError(f'no safetensors weights in base export: {export}')
        self.cache = {}

    def alpha(self, name, shape):
        if name not in self.cache:
            key = name + '.weight'
            if self.compact is not None:
                weight = self.compact.weight(key).to(dtype=torch.bfloat16, device='cpu')
            elif key not in self.weights:
                raise ValueError(f'base export missing linear: {key}')
            else:
                shard = (self.export / self.weights[key]).resolve()
                if not shard.is_relative_to(self.export.resolve()):
                    raise ValueError('base shard outside export directory')
                with self.safe_open(str(shard), framework='pt', device='cpu') as reader:
                    weight = reader.get_tensor(key).to(dtype=torch.bfloat16, device='cpu')
            if list(weight.shape) != list(shape):
                raise ValueError(f'{name}: base weight shape differs')
            # Invoke the harness's exact row-alpha implementation. Rank zero
            # avoids allocating dense FSA masks/masters or LoRA factors.
            with torch.no_grad():
                alpha = harness().LoRAParam(weight, 0, ternary=True).alpha.squeeze(1).clone()
            self.cache[name] = (tuple(shape), alpha)
        saved_shape, alpha = self.cache[name]
        if saved_shape != tuple(shape):
            raise ValueError(f'{name}: base weight shape differs')
        return alpha


def base_alphas(base_export):
    return base_export if isinstance(base_export, BaseAlphas) else BaseAlphas(base_export)


def derived_alphas(state, base_export):
    base = base_alphas(base_export)
    derived = {}
    for name, entry in state['entries'].items():
        if 'alpha' not in entry:
            continue
        expected = base.alpha(name, entry['shape'])
        try:
            assert_identical(entry['alpha'], expected, name + '.alpha')
        except AssertionError:
            continue
        derived[name] = dict(alpha='derived', alpha_sha256=hashlib.sha256(tensor_bytes(expected)).hexdigest())
    return derived


def resolve_patch(path):
    """Accept a final.pt, explicit run directory, or harness default *_patches."""
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (path/'final.pt', path.with_name(path.name+'_patches')/'final.pt'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'no saved patch at {path} or its default _patches directory')


def varints(values):
    out = bytearray()
    for value in values:
        value = int(value)
        if value < 0:
            raise ValueError('negative position delta')
        while value >= 128:
            out.append((value & 127) | 128)
            value >>= 7
        out.append(value)
    return bytes(out)


def unvarints(data):
    values, value, shift = [], 0, 0
    for byte in data:
        value |= (byte & 127) << shift
        if byte < 128:
            values.append(value)
            value, shift = 0, 0
        else:
            shift += 7
            if shift > 63:
                raise ValueError('invalid varint')
    if shift:
        raise ValueError('truncated varint')
    return np.asarray(values, dtype=np.int64)


def deltas(ids):
    return varints(np.diff(ids, prepend=0))


def signs(values):
    return np.packbits(values > 0, bitorder='little').tobytes()


def unsigns(data, n):
    return (np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='little')[:n].astype(np.int8)*2-1)


def pack_trits(values):
    # Five base-3 digits per byte (243 states), including zeros.
    digits = np.pad(values.astype(np.int16)+1, (0, (-len(values)) % 5), constant_values=1)
    return (digits.reshape(-1, 5) @ (3**np.arange(5))).astype(np.uint8).tobytes()


def unpack_trits(data, n):
    values = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    if (values >= 243).any():
        raise ValueError('invalid packed trits')
    return ((values[:, None] // (3**np.arange(5))) % 3).ravel()[:n].astype(np.int8)-1


def sparse_payload(entry, name, encoding):
    idx, val = entry['idx'], entry['val']
    if (entry.get('kind') != 'fsa' or entry.get('fmt') != 'ternary' or
            idx.dtype not in (torch.int32, torch.int64) or val.dtype != torch.int8 or
            idx.ndim != 1 or val.shape != idx.shape or len(entry['shape']) != 2):
        raise ValueError('requires sparse ternary FSA entries')
    ids, values = idx.cpu().numpy().astype(np.int64), val.cpu().numpy()
    rows, cols = entry['shape']
    if ((ids < 0).any() or (ids >= rows*cols).any() or (np.diff(ids) <= 0).any() or
            not np.isin(values, [-1, 1]).all()):
        raise ValueError('positions must be sorted unique and signs nonzero ternary')
    descriptor = dict(n=len(ids), idx_dtype=str(idx.dtype).split('.')[-1], encoding=encoding)
    if encoding == 'coordinate':
        return descriptor, [deltas(ids), signs(values)]
    if encoding == 'block':
        blocks, inverse = np.unique(ids//8, return_inverse=True)
        masks = np.zeros(len(blocks), dtype=np.uint8)
        np.bitwise_or.at(masks, inverse, (1 << (ids % 8)).astype(np.uint8))
        return descriptor, [deltas(blocks), masks.tobytes(), signs(values)]
    if encoding != 'whole-unit':
        raise ValueError('unknown encoding')
    axis = 1 if name.endswith('down_proj') else 0
    unit_ids = ids % cols if axis else ids//cols
    order = unit_ids.argsort(kind='stable')
    unit_ids, ordered_ids, ordered_values = unit_ids[order], ids[order], values[order]
    units, starts = np.unique(unit_ids, return_index=True)
    stops = np.append(starts[1:], len(ids))
    width = rows if axis else cols
    # One vector at a time: bounded temporary memory, even for large experts.
    payload = bytearray()
    for unit, start, stop in zip(units, starts, stops):
        dense = np.zeros(width, dtype=np.int8)
        positions = ordered_ids[start:stop]
        dense[positions//cols if axis else positions % cols] = ordered_values[start:stop]
        payload.extend(pack_trits(dense))
    descriptor.update(axis=axis, width=width)
    return descriptor, [deltas(units), bytes(payload)]


def decode_sparse(desc, chunks, shape):
    enc, n = desc['encoding'], desc['n']
    ids = np.cumsum(unvarints(chunks[0]))
    if enc == 'coordinate':
        values = unsigns(chunks[1], n)
    elif enc == 'block':
        masks = np.frombuffer(chunks[1], dtype=np.uint8)
        if len(masks) != len(ids):
            raise ValueError('invalid block masks')
        ids = (ids[:, None]*8 + np.arange(8))[((masks[:, None] >> np.arange(8)) & 1).astype(bool)]
        values = unsigns(chunks[2], n)
    elif enc == 'whole-unit':
        width, axis = desc['width'], desc['axis']
        size = math.ceil(width/5)
        if len(chunks[1]) != len(ids)*size:
            raise ValueError('invalid unit payload')
        positions, codes = [], []
        for i, unit in enumerate(ids):
            dense = unpack_trits(chunks[1][i*size:(i+1)*size], width)
            live = np.flatnonzero(dense)
            positions.append(live*shape[1]+unit if axis else unit*shape[1]+live)
            codes.append(dense[live])
        ids = np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
        values = np.concatenate(codes) if codes else np.empty(0, dtype=np.int8)
        order = ids.argsort()
        ids, values = ids[order], values[order]
    else:
        raise ValueError('unknown sparse encoding')
    if len(ids) != n or len(values) != n:
        raise ValueError('invalid sparse count')
    return torch.tensor(ids, dtype=getattr(torch, desc['idx_dtype'])), torch.tensor(values, dtype=torch.int8)


def encode_patch(state, encoding, base_export=None, *, _derived=None):
    if state.get('format') != 'e2e-fsa-patch-v1':
        raise ValueError('unsupported harness patch format')
    payload = bytearray()
    derived = (_derived if _derived is not None else
               derived_alphas(state, base_export) if base_export is not None else {})
    def section(data):
        start = len(payload)
        payload.extend(data)
        return [start, len(data)]
    def tree(obj):
        if torch.is_tensor(obj):
            tensor = obj.detach().cpu().contiguous()
            data = tensor_bytes(tensor)
            return {'tensor': str(tensor.dtype).split('.')[-1], 'shape': list(tensor.shape), 'section': section(data)}
        if isinstance(obj, dict):
            return {'dict': [[tree(k), tree(v)] for k, v in obj.items()]}
        if isinstance(obj, (list, tuple)):
            return {type(obj).__name__: [tree(v) for v in obj]}
        if type(obj) is float:
            return {'float64': struct.pack('<d', obj).hex()}
        if obj is None or isinstance(obj, (str, bool, int)):
            return {'scalar': obj}
        raise ValueError(f'unsupported metadata type {type(obj)}')
    entries = {}
    for name, entry in state['entries'].items():
        desc, chunks = sparse_payload(entry, name, encoding)
        desc['sections'] = [section(chunk) for chunk in chunks]
        omit = ('idx', 'val', 'alpha') if name in derived else ('idx', 'val')
        entries[name] = dict(sparse=desc, rest=tree({k: v for k, v in entry.items() if k not in omit}))
        entries[name].update(derived.get(name, {}))
    manifest = dict(encoding=encoding, rest=tree({k: v for k, v in state.items() if k != 'entries'}), entries=entries,
                    payload_sha256=hashlib.sha256(payload).hexdigest())
    header = json.dumps(manifest, separators=(',', ':'), allow_nan=True).encode()
    return MAGIC + struct.pack('<Q', len(header)) + header + bytes(payload)


def decode_patch(data, base_export=None):
    if data[:8] != MAGIC or len(data) < 16:
        raise ValueError('invalid packed patch')
    length = struct.unpack('<Q', data[8:16])[0]
    manifest = json.loads(data[16:16+length])
    payload = data[16+length:]
    if hashlib.sha256(payload).hexdigest() != manifest['payload_sha256']:
        raise ValueError('packed patch checksum mismatch')
    def section(spec):
        start, size = spec
        if start < 0 or size < 0 or start+size > len(payload):
            raise ValueError('invalid binary section')
        return payload[start:start+size]
    def tree(obj):
        if 'tensor' in obj:
            raw = section(obj['section'])
            dtype = getattr(torch, obj['tensor'])
            return (torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(obj['shape']) if raw else
                    torch.empty(obj['shape'], dtype=dtype))
        if 'dict' in obj:
            return {tree(k): tree(v) for k, v in obj['dict']}
        for kind in ('list', 'tuple'):
            if kind in obj:
                values = [tree(v) for v in obj[kind]]
                return tuple(values) if kind == 'tuple' else values
        if 'float64' in obj:
            return struct.unpack('<d', bytes.fromhex(obj['float64']))[0]
        return obj['scalar']
    state = tree(manifest['rest'])
    state['entries'] = {}
    base = None
    for name, record in manifest['entries'].items():
        entry = tree(record['rest'])
        if record.get('alpha') == 'derived':
            if base_export is None:
                raise ValueError('derived alpha requires base_export')
            if base is None:
                base = base_alphas(base_export)
            alpha = base.alpha(name, entry['shape'])
            if hashlib.sha256(tensor_bytes(alpha)).hexdigest() != record['alpha_sha256']:
                raise ValueError(f'{name}: derived alpha checksum mismatch')
            entry['alpha'] = alpha.clone()
        desc = record['sparse']
        entry['idx'], entry['val'] = decode_sparse(desc, [section(s) for s in desc['sections']], entry['shape'])
        state['entries'][name] = entry
    return state


def assert_identical(left, right, path='patch'):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype != right.dtype or left.shape != right.shape or (left.numel() and not torch.equal(
                left.cpu().contiguous().reshape(-1).view(torch.uint8), right.cpu().contiguous().reshape(-1).view(torch.uint8))):
            raise AssertionError(f'{path}: tensor changed')
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f'{path}: keys changed')
        for key in left:
            assert_identical(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (tuple, list)):
        if type(left) != type(right) or len(left) != len(right):
            raise AssertionError(f'{path}: sequence changed')
        for i, (a, b) in enumerate(zip(left, right)):
            assert_identical(a, b, f'{path}[{i}]')
    elif type(left) is float:
        if type(right) is not float or struct.pack('<d', left) != struct.pack('<d', right):
            raise AssertionError(f'{path}: metadata float changed')
    elif type(left) != type(right) or left != right:
        raise AssertionError(f'{path}: metadata changed')


def component_bytes(data):
    """Disjoint byte counts, summing to the complete container length.

    Block occupancy masks are positions. Whole-unit trits combine occupancy
    and signs; their indivisible payload is counted under signs.
    """
    length = struct.unpack('<Q', data[8:16])[0]
    manifest = json.loads(data[16:16+length])
    positions = signs_size = 0
    for record in manifest['entries'].values():
        desc = record['sparse']
        chunks = desc['sections']
        positions += chunks[0][1]
        if desc['encoding'] == 'block':
            positions += chunks[1][1]
        signs_size += chunks[-1][1]
    header = 16 + length
    return dict(positions=positions, signs=signs_size,
                aux=len(data)-header-positions-signs_size, header=header)


def written_facts(meta):
    if meta.get('c37_facts'):
        E = harness()
        facts = [E.Fact(**row) for row in meta['c37_facts']]
        return len(E.person_indices(facts, meta.get('train_person'),
                                   meta.get('train_attr_group'), meta.get('attr_groups', 1)))
    return meta.get('facts_per_person', 0)*(1 if meta.get('train_person') is not None else meta.get('people', 1))


def measure_patch(path, out_dir, result_path=None, base_export=None):
    path, out_dir = resolve_patch(path), Path(out_dir)
    state = torch.load(path, map_location='cpu', weights_only=False)
    base = base_alphas(base_export) if base_export is not None else None
    derived = derived_alphas(state, base) if base is not None else {}
    candidates, reports = {}, {}
    for encoding in ENCODINGS:
        stored = encode_patch(state, encoding)
        data = encode_patch(state, encoding, _derived=derived) if base is not None else stored
        assert_identical(state, decode_patch(stored))
        assert_identical(state, decode_patch(data, base))
        candidates[encoding] = data
        reports[encoding] = dict(bytes=len(data), ratio=path.stat().st_size/len(data),
                                 status='keep' if len(data)*2 <= path.stat().st_size else 'kill: saves <2x', bit_identical=True,
                                 bytes_alpha_stored=len(stored), bytes_alpha_derived=len(data) if base is not None else None,
                                 components_alpha_stored=component_bytes(stored),
                                 components_alpha_derived=component_bytes(data) if base is not None else None)
    chosen = min(candidates, key=lambda enc: len(candidates[enc]))
    stored_chosen = min(reports, key=lambda enc: reports[enc]['bytes_alpha_stored'])
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir/(path.parent.name+'-'+path.stem+'.fsa57')
    target.write_bytes(candidates[chosen])
    assert_identical(state, decode_patch(target.read_bytes(), base))
    if result_path is not None and not Path(result_path).is_file():
        raise FileNotFoundError(result_path)
    result_path = Path(result_path) if result_path else path.parent.parent/(path.parent.name.removesuffix('_patches')+'.json')
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    storage = result.get('storage') or {}
    estimate = result.get('patch_bytes')
    if estimate is None:
        estimate = storage.get('fsa', storage).get('bytes')
    meta = state['meta']
    facts = written_facts(meta)
    # Realized retained facts are supplied by the caller; never infer from paraphrases.
    return dict(source=str(path), output=str(target), encoding=chosen, packed_bytes=target.stat().st_size,
                final_pt_bytes=path.stat().st_size, harness_estimate_bytes=estimate,
                packed_vs_harness_estimate=target.stat().st_size/estimate if estimate else None,
                harness_estimate_status='recorded result' if estimate is not None else 'unavailable: supply sibling result JSON',
                written_facts=facts, candidates=reports, promotable=reports[chosen]['status']=='keep',
                base_export=str(base.export) if base is not None else None,
                derived_alpha_entries=len(derived), stored_alpha_entries=sum('alpha' in e for e in state['entries'].values())-len(derived),
                encoding_alpha_stored=stored_chosen,
                bytes_alpha_stored=reports[stored_chosen]['bytes_alpha_stored'],
                bytes_alpha_derived=reports[chosen]['bytes_alpha_derived'],
                components_alpha_stored=reports[stored_chosen]['components_alpha_stored'],
                components_alpha_derived=reports[chosen]['components_alpha_derived'],
                components=component_bytes(candidates[chosen]))


def marginal(rows, facts):
    if len(rows) != len(facts) or any(n < 0 or not math.isfinite(n) for n in facts):
        raise ValueError('one nonnegative fact count per patch required')
    out = []
    for a, b, fa, fb in zip(rows, rows[1:], facts, facts[1:]):
        delta = fb-fa
        out.append(dict(from_patch=a['source'], to_patch=b['source'], delta_facts=delta,
                        delta_bytes=b['packed_bytes']-a['packed_bytes'],
                        bytes_per_added_fact=(b['packed_bytes']-a['packed_bytes'])/delta if delta else None,
                        final_pt_delta_bytes=b['final_pt_bytes']-a['final_pt_bytes']))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patches', type=Path, nargs='+', required=True, help='final.pt paths or run directories (including default *_patches)')
    p.add_argument('--retained-facts', type=float, nargs='+', help='one distinct retained-fact count per patch')
    p.add_argument('--results', type=Path, nargs='+', help='one harness result JSON per patch (default sibling JSON)')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-export', type=Path, help='Compact HF or safetensors base export; derive exactly matching alpha tensors')
    a = p.parse_args(argv)
    if a.results is not None and len(a.results) != len(a.patches):
        raise ValueError('one result JSON per patch required')
    if a.retained_facts is not None and len(a.retained_facts) != len(a.patches):
        raise ValueError('one retained-fact count per patch required')
    base = BaseAlphas(a.base_export) if a.base_export is not None else None
    rows = [measure_patch(path, a.out_dir, result, base) for path, result in
            zip(a.patches, a.results or [None]*len(a.patches))]
    counts = a.retained_facts if a.retained_facts is not None else [r['written_facts'] for r in rows]
    report = dict(patches=rows, marginal=marginal(rows, counts),
                  fact_denominator='retained facts supplied' if a.retained_facts is not None else 'written facts; retention not measured')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print('SERIALIZE ' + json.dumps(report))
    return report


if __name__ == '__main__':
    main()
