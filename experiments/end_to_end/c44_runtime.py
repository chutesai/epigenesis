"""Training recovery and process-wide, progress-driven heartbeats (no RNG draws)."""
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import torch


class Heartbeat:
    def __init__(self, interval=120., clock=time.monotonic, emit=print):
        self.interval, self.clock, self.emit = interval, clock, emit
        self.started = self.last = clock()

    def __call__(self, phase, done, total, force=False):
        now = self.clock()
        if not force and now - self.last < self.interval:
            return False
        self.emit('HEARTBEAT ' + json.dumps(dict(phase=phase, done=done, total=total,
                                               t=now-self.started)), flush=True)
        self.last = now
        return True


heartbeat = Heartbeat()


def progress(iterable, phase, total=None):
    if total is None and hasattr(iterable, '__len__'):
        total = len(iterable)
    for done, item in enumerate(iterable, 1):
        yield item
        heartbeat(phase, done, total)


def configuration(args):
    # Compare parsed options, so --flag=x and --flag x have the same identity.
    config = {k: v for k, v in vars(args).items()
              if k not in ('resume', 'out', 'ckpt_every', 'ckpt_minutes', 'patch_dir') and not k.startswith('_')}
    for key, value in config.items():
        if key in INPUT_PATHS or isinstance(value, Path):
            if isinstance(value, (list, tuple)):
                config[key] = [str(Path(p).resolve()) for p in value]
            elif value:
                config[key] = str(Path(value).resolve())
        elif key == 'extra_eval_sets' and value:
            config[key] = ','.join(str(Path(p.strip()).resolve()) for p in value.split(',') if p.strip())
    config.setdefault('eda_backend', 'naive_fp32')
    return config


INPUT_PATHS = ('anchor', 'eval_set', 'teacher_patch', 'sibling_patches', 'load_patch',
               'resident_patch', 'resident_mask_only', 'export', 'mask_compare',
               'rehearsal_set', 'calib_set', 'route_kl_set', 'unit_burst_replay', 'bench_dir', 'in_topic_panel')


def input_fingerprints(args):
    """Stream content hashes; absent optional/default files remain part of identity."""
    paths = []
    for key in INPUT_PATHS:
        if key in ('export', 'bench_dir'):
            continue
        value = getattr(args, key, None)
        paths.extend(value if isinstance(value, (list, tuple)) else [value] if value else [])
    paths.extend(p.strip() for p in (getattr(args, 'extra_eval_sets', None) or '').split(',') if p.strip())
    export = getattr(args, 'export', None)
    if export:
        paths.append(Path(export) / 'manifest.json')
    # The external builder's manifest schema/content-hash guarantees are unknown.
    # Hash every export file (including nested weight shards and relay manifests)
    # rather than trust the top-level manifest to identify its payload contents.
    # The file inventory also detects additions/removals with an unchanged manifest.
    for directory in (export, getattr(args, 'bench_dir', None)):
        if directory:
            root = Path(directory)
            paths.append(root)
            paths.extend(sorted(p for p in root.rglob('*') if p.is_file()))
    hashes = {}
    hashed_bytes = 0
    for value in paths:
        path = Path(value).resolve()
        digest = hashlib.sha256()
        if path.is_file():
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    hashed_bytes += len(block)
                    heartbeat('input_fingerprints', hashed_bytes, None)
            hashes[str(path)] = digest.hexdigest()
        else:
            hashes[str(path)] = None
    return hashes


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


_save_sink = ContextVar('c44_save_sink', default=None)


@contextmanager
def forward_saves(path, sink):
    token = _save_sink.set((path, sink))
    try:
        yield
    finally:
        _save_sink.reset(token)


class Checkpoints:
    def __init__(self, args):
        if args.ckpt_every < 0 or not math.isfinite(args.ckpt_minutes) or args.ckpt_minutes < 0:
            raise SystemExit('--ckpt-every and --ckpt-minutes must be finite and >= 0')
        self.config = configuration(args)
        self.hash = config_hash(self.config)
        root = args.patch_dir or str(Path(args.out).with_suffix('')) + '_patches'
        self.path = Path(root) / 'resume.pt'
        self.every, self.minutes = args.ckpt_every, args.ckpt_minutes
        self.enabled = bool(self.every or self.minutes or args.resume)
        self.inputs = input_fingerprints(args) if self.enabled else {}
        self.peak_alloc_gb = 0.
        self.last_step, self.last_time = 0, time.monotonic()
        self.saved = None
        if args.resume and self.path.exists():
            self.saved = torch.load(self.path, map_location='cpu', weights_only=False)
            if self.saved.get('version') != 1 or self.saved.get('config_hash') != self.hash:
                previous = self.saved.get('config', {})
                diff = {k: (previous.get(k), self.config.get(k)) for k in previous.keys() | self.config.keys()
                        if previous.get(k) != self.config.get(k)}
                raise SystemExit(f'--resume config mismatch (saved, current): {diff}; checkpoint {self.path}')
            if self.saved.get('input_sha256') != self.inputs:
                raise SystemExit(f'--resume input contents changed; checkpoint {self.path}')
            self.peak_alloc_gb = self.saved.get('max_alloc_gb', 0.)
            self.last_step = self.saved['step']

    def due(self, step):
        return self.enabled and ((self.every and step-self.last_step >= self.every) or
                                 (self.minutes and time.monotonic()-self.last_time >= 60*self.minutes))

    def save(self, step, **payload):
        state = dict(version=1, config=self.config, config_hash=self.hash, input_sha256=self.inputs,
                     max_alloc_gb=self.memory_peak(), step=step, **payload)
        sink = _save_sink.get()
        if sink is not None and sink[0] == self.path:
            sink[1](state)
        else:
            atomic_save(self.path, state)
        self.last_step, self.last_time = step, time.monotonic()

    def memory_peak(self):
        # Allocator telemetry is excluded from numerical exactness; preserve its running peak.
        current = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.
        self.peak_alloc_gb = max(self.peak_alloc_gb, current)
        return self.peak_alloc_gb


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise RuntimeError('resume requires the same CUDA device count')
        torch.cuda.set_rng_state_all(state['cuda'])


def _portable(value):
    """State values only: never pickle model references, hooks or closures."""
    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor, random.Random, torch.Generator)):
        return True
    if isinstance(value, dict):
        return all(_portable(k) and _portable(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return all(_portable(v) for v in value)
    return False


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        out = copy.copy(value)
        for key, item in value.items():
            out[key] = _snapshot(item)
        return out
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def object_state(obj):
    if obj is None:
        return None
    if isinstance(obj, random.Random):
        return {'python_rng': obj.getstate()}
    if isinstance(obj, torch.Generator):
        return {'torch_rng': obj.get_state()}
    # Rebuilt module references/coordinates/hooks; noise deltas are overwritten by draw().
    benign = {'UnitBurst': {'banks', 'coords', 'names', 'hooks'},
              'MatchedActivationNoise': {'deltas', 'hooks'}}.get(type(obj).__name__, set())
    # Module registries belong to state_dict/reconstructed topology, not plain attrs.
    if isinstance(obj, torch.nn.Module):
        benign = benign | {key for key in vars(torch.nn.Module()) if key.startswith('_')}
    state = {}
    for key, value in vars(obj).items():
        if key in benign:
            continue
        if not _portable(value):
            raise TypeError(f'non-portable resume state: {type(obj).__name__}.{key}')
        state[key] = _snapshot(value)
    return state


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        out = copy.copy(value)
        for k, v in value.items():
            out[k] = _move(v, device)
        return out
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    return value


def restore_object(obj, state, device):
    if obj is None:
        if state is not None:
            raise RuntimeError('resume object topology differs')
        return
    if isinstance(obj, random.Random):
        obj.setstate(state['python_rng'])
        return
    if isinstance(obj, torch.Generator):
        obj.set_state(state['torch_rng'])
        return
    for key, value in state.items():
        current = getattr(obj, key, None)
        if isinstance(current, (random.Random, torch.Generator)):
            current.setstate(value.getstate()) if isinstance(current, random.Random) else current.set_state(value.get_state())
        elif isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
            with torch.no_grad():
                value = value.to(current.device)
                if any(n > 1 and stride == 0 for n, stride in zip(current.shape, current.stride())):
                    if not torch.equal(current, value):
                        raise RuntimeError(f'resume expanded constant differs: {key}')
                else:
                    current.copy_(value)
        elif isinstance(value, (str, bool, int, float, type(None))) and current == value:
            pass  # Keep reconstructed constants and their sharing with export metadata.
        else:
            setattr(obj, key, _move(value, device))


def training_state(model, module_types, optimizer, values, objects):
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            modules[name] = dict(weights=module.state_dict(), attributes=object_state(module),
                                 gradients={key: None if p.grad is None else p.grad.detach().to('cpu', copy=True)
                                            for key, p in module.named_parameters()})
    # values is stored by reference: torch.save and forward_saves must remain synchronous.
    return dict(modules=modules, optimizer=optimizer.state_dict(), optimizer_type=type(optimizer).__name__,
                values=values, objects={key: object_state(obj) for key, obj in objects.items()}, rng=rng_state())


def restore_training(state, model, module_types, optimizer, objects, device):
    modules = {name: m for name, m in model.named_modules() if isinstance(m, module_types)}
    if modules.keys() != state['modules'].keys():
        raise RuntimeError('resume adapter/resident topology differs')
    for name, module in modules.items():
        saved = state['modules'][name]
        module.load_state_dict(saved['weights'])
        restore_object(module, saved['attributes'], device)
        for key, param in module.named_parameters():
            grad = saved['gradients'][key]
            param.grad = None if grad is None else grad.to(param.device)
    optimizer.load_state_dict(state['optimizer'])
    for key, obj in objects.items():
        restore_object(obj, state['objects'][key], device)
    # Setup and baseline evaluation can draw random numbers. Restore only afterwards.
    restore_rng(state['rng'])
    return state['values']
