"""Bounded-memory uint32 token sources and exact, hashed 32-gram holdout checks."""
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .c44_runtime import heartbeat
except ImportError:
    from c44_runtime import heartbeat


def read_tokens(path):
    path = Path(path)
    if path.suffix == '.json':
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or any(type(x) is not int or not 0 <= x < 2**32 for x in rows):
            raise ValueError(f'{path}: expected a list of uint32 token IDs')
        return np.asarray(rows, dtype='<u4')
    if path.stat().st_size % 4:
        raise ValueError(f'{path}: uint32 file size is not divisible by four')
    return np.memmap(path, mode='r', dtype='<u4')


def grams(tokens, stride=1, chunk=65536):
    # Keep only one chunk's bytes; include n-grams crossing chunk/window boundaries.
    for start in range(0, max(0, len(tokens)-31), chunk):
        stop = min(start+chunk, len(tokens)-31)
        data = np.asarray(tokens[start:stop+31], dtype='<u4').tobytes()
        for i in range((-start) % stride, stop-start, stride):
            yield start+i, hashlib.blake2b(data[4*i:4*(i+32)], digest_size=16).digest()
        heartbeat('overlap_scan', stop, len(tokens))


class OverlapGuard:
    """Same denominator as e2e ngram_overlap: sampled reference grams hit by source.

    Scan every source gram, including context boundaries. 128-bit BLAKE2 hashes
    bound reference memory without retaining the training stream's gram set.
    """
    def __init__(self, references, reference_stride=16):
        self.refs = {str(name): [g for _, g in grams(tokens, stride=reference_stride)]
                     for name, tokens in references.items()}
        self.wanted = set(g for group in self.refs.values() for g in group)

    def scan(self, sources):
        hits = set()
        for tokens in sources:
            if self.wanted:
                hits.update(g for _, g in grams(tokens) if g in self.wanted)
        return hits

    def fractions(self, hits):
        return {name: sum(g in hits for g in group)/max(len(group), 1)
                for name, group in self.refs.items()}

    def check(self, sources, source='stream', limit=.01):
        report = self.fractions(self.scan(sources))
        print('OVERLAP ' + json.dumps(dict(source=source, fractions=report)), flush=True)
        if max(report.values(), default=0.) > limit:
            raise ValueError(f'{source} 32-gram overlap exceeds {limit:.0%}: {report}')
        return report


def window_overlaps(windows, sources):
    """Strict disjointness: scan EVERY gram of large training sources against
    EVERY gram of the small candidate panel. Memory scales with the panel only.
    """
    guard = OverlapGuard({str(i): w for i,w in enumerate(windows)}, reference_stride=1)
    contaminated = set()
    report = {}
    for name, tokens in sources.items():
        fractions = guard.fractions(guard.scan([tokens]))
        matched = [int(i) for i,value in fractions.items() if value > 0]
        contaminated.update(matched)
        report[name] = dict(matched_windows=matched, windows=len(windows))
    return contaminated, report
