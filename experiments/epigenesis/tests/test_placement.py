"""CPU contracts for placement adapters, inventory, training and interventions."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# mesh source: local checkout next to this repo, or the GPU box's copy
MESH_ROOT = next((str(p) for p in (Path(__file__).resolve().parents[4] / 'mesh', Path('/workspace/malleable/mlf'))
                  if (p / 'mesh').is_dir()), str(Path(__file__).resolve().parents[4] / 'mesh'))
import placement as p
from make_user import build_corpus
from test_ablate import OffsetTokenizer


class Expert(nn.Module):
    def __init__(self, d=4, h=6, device=None):
        super().__init__()
        self.up_proj = nn.Linear(d, h, bias=False, device=device)
        self.down_proj = nn.Linear(h, d, bias=False, device=device)

    def forward(self, x):
        return self.down_proj(self.up_proj(x).relu())


class MoE(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.experts = nn.ModuleList([Expert(device=device), Expert(device=device)])
        self.shared_experts = nn.ModuleList([Expert(device=device)])
        self.router = nn.Linear(4, 2, device=device)
        self.latent_down = nn.Linear(4, 4, device=device)

    def forward(self, x):
        return x + sum(e(x) for e in self.experts) + self.shared_experts[0](x)


class EDALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self._fast_impl = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 4))
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        return x + self._fast_impl(self.norm(x))


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, nn.Linear(4, 4))

    def forward(self, x):
        return x + self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class MSAAttentionLayer(AttentionLayer):
    def __init__(self):
        super().__init__()
        self.index = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 4))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 4)
        self.layers = nn.ModuleList([EDALayer(), MoE(), AttentionLayer(), MoE(), MSAAttentionLayer(), MoE()])
        self.lm_head = nn.Linear(4, 256)

    def forward(self, x):
        h = self.embed(x)
        for block in self.layers:
            h = block(h)
        return self.lm_head(h)


@pytest.fixture
def inventory():
    model = Model()
    targets, families = p.enumerate_linears(model)
    return model, targets, families


def test_inventory_exclusions_nested_projections_and_block_ids(inventory):
    model, targets, families = inventory
    assert len(targets) == 28  # 18 expert linears + 2 EDA + 4 attention + 4 MSA.
    names = [t.name for t in targets]
    assert 'layers.0._fast_impl.0' in names
    assert not any('index' in n for n in names)
    assert 'layers.1.shared_experts.0.down_proj' in names
    assert all(not any(word in name for word in ('router', 'latent_down', 'norm', 'embed', 'lm_head')) for name in names)
    assert [f['layer'] for f in families if f['has_moe']] == [1, 3, 5]
    assert [f['mixer'] for f in families] == ['EDALayer', 'none', 'AttentionLayer', 'none', 'MSAAttentionLayer', 'none']
    assert sum(p.RANK * sum(t.linear.weight.shape) for t in targets) == p.parameter_counts(targets)['all']['parameters']


def test_exact_count_includes_shared_and_nested_mixers(inventory):
    _, targets, _ = inventory
    counts = p.parameter_counts(targets)
    assert counts['experts']['parameters'] == 18 * 16 * 10
    assert counts['mixers']['parameters'] == 16 * (2 * 7 + 8 * 8)
    assert counts['all']['parameters'] == 4128
    assert counts['all']['params_adam_bytes'] == 4128 * 12


def test_32_layer_128_expert_count_and_guard_without_allocating():
    # Full-width hypothetical experts exceed the guard; actual latent experts do not.
    with torch.device('meta'):
        full = [p.Target(str(i), i // 256, 'experts', nn.Linear(1152, 2304)) for i in range(32 * 128 * 2)]
        latent = [p.Target(str(i), i // 256, 'experts', nn.Linear(384, 2304)) for i in range(32 * 128 * 2)]
        shared = [p.Target('s' + str(i), i // 2, 'experts', nn.Linear(1152, 2304)) for i in range(32 * 2)]
        oversized = [p.Target(str(i), i // 256, 'experts', nn.Linear(1152, 4096)) for i in range(32 * 128 * 2)]
    assert p.parameter_counts(full)['experts']['parameters'] == 452984832
    assert p.check_memory(full) == 452984832  # 5.0625 GiB, below 6 GiB.
    assert p.check_memory(latent + shared) == 355860480
    with pytest.raises(ValueError, match='limit 6 GiB'):
        p.check_memory(oversized)
    with pytest.raises(ValueError, match='--targets mixers'):
        p.check_memory(full, limit=5 * 2**30)
    n = p.parameter_counts(full)['all']['parameters']
    assert p.check_memory(full, limit=n * 12) == n
    with pytest.raises(ValueError, match=f'{n:,}'):
        p.check_memory(full, limit=n * 12 - 1)


def test_guard_precedes_factor_allocation(inventory, monkeypatch):
    model, targets, _ = inventory
    monkeypatch.setattr(p, 'check_memory', lambda targets: (_ for _ in ()).throw(ValueError('too big')))
    monkeypatch.setattr(p, 'LoRA', lambda weight: pytest.fail('must not allocate'))
    with pytest.raises(ValueError, match='too big'):
        p.attach(model, targets)
    assert all(not p.parametrize.is_parametrized(t.linear) for t in targets)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_lora_initial_output_scaling_gradient_and_disable(dtype):
    lin = nn.Linear(4, 6, bias=False, dtype=dtype)
    base = lin.weight.detach().clone()
    m = p.LoRA(base)
    lin.requires_grad_(False)
    p.parametrize.register_parametrization(lin, 'weight', m)
    assert m.A.dtype == m.B.dtype == torch.float32
    assert not m.B.any() and m.A.any()
    assert torch.equal(lin.weight, base)
    lin(torch.ones(1, 4, dtype=dtype)).float().sum().backward()
    assert m.B.grad.any() and not m.A.grad.any()
    assert lin.parametrizations.weight.original.grad is None
    with torch.no_grad():
        m.B.fill_(1)
    assert torch.equal(lin.weight, base + (m.B @ m.A).to(dtype))
    m.enabled = False
    assert torch.equal(lin.weight, base)


def test_zeroed_restores_state_on_exception(inventory):
    model, targets, _ = inventory
    mods = p.attach(model, targets)
    names = list(mods)[:2]
    mods[names[1]].enabled = False
    with pytest.raises(RuntimeError):
        with p.zeroed(mods, names):
            assert not any(mods[n].enabled for n in names)
            raise RuntimeError()
    assert mods[names[0]].enabled and not mods[names[1]].enabled


def test_bands_partition_actual_block_indices(inventory):
    _, targets, families = inventory
    groups = p.attribution_groups(targets, families)
    assert len(groups) == 11
    bands = groups[6:9]
    assert [b['layers'] for b in bands] == [[0, 1], [2, 3], [4, 5]]
    assert sorted(n for b in bands for n in b['names']) == sorted(t.name for t in targets)
    assert {n for b in groups[-2:] for n in b['names']} == {t.name for t in targets}


def test_attribute_interventions_and_markdown(inventory, monkeypatch):
    model, targets, families = inventory
    mods = p.attach(model, targets)
    seen = []
    def acc(*args):
        disabled = [n for n, m in mods.items() if not m.enabled]
        seen.append(disabled)
        return dict(acc=1 - len(disabled) / len(mods), n=400, se=0.)
    monkeypatch.setattr(p, 'dev_accuracy', acc)
    result = p.attribute(model, None, 'cpu', [], targets, families, mods)
    assert not seen[0]
    assert seen[1:] == [g['names'] for g in p.attribution_groups(targets, families)]
    assert all(m.enabled for m in mods.values())
    assert result['attribution'][1]['drop'] == pytest.approx(6 / 28)
    assert '| layer_1 | none | True |' in p.markdown(result)


@pytest.mark.parametrize('family', ['ticket', 'qa'])
def test_corrupt_key_only_different_action_and_unique_match(family):
    corpus = build_corpus(rules='coarse')
    ids = [p['id'] for p in p.a.trained_policies(corpus, 1)]
    probes = p.a.large_dev_concepts(corpus, ids, 40, corpus['seed'])
    for probe in [x for x in probes if x['family'] == family]:
        before = copy.deepcopy(probe)
        corrupt, swap = p.corrupt_probe(probe, corpus['policies'])
        assert probe == before
        assert {k for k in probe['case'] if probe['case'][k] != corrupt['case'][k]} == {swap['attribute']}
        clean = next(x for x in corpus['policies'] if x['id'] == probe['policy_id'])
        matches = [x for x in corpus['policies'] if p.a.case_matches(x, corrupt['case'])]
        assert len(matches) == 1 and matches[0]['action'] != clean['action']
        assert corrupt['options'] == probe['options'] and corrupt['answer_idx'] == probe['answer_idx']


def test_corruption_rejects_no_alternative():
    corpus = build_corpus(rules='coarse')
    policy = p.a.trained_policies(corpus, 1)[0]
    probe = p.a.large_dev_concepts(corpus, [policy['id']], 2, corpus['seed'])[0]
    with pytest.raises(ValueError, match='No unique'):
        p.corrupt_probe(probe, [policy])


@pytest.mark.parametrize('output_type', [torch.Tensor, tuple, list])
def test_replace_hidden_preserves_auxiliary(output_type):
    h = torch.zeros(1, 3, 4)
    clean = torch.ones_like(h)
    output = h if output_type is torch.Tensor else output_type([h, 'aux'])
    restored = p.replace_hidden(output, clean)
    assert torch.equal(p.hidden(restored), clean)
    if output_type is not torch.Tensor:
        assert restored[1] == 'aux' and isinstance(restored, output_type)
    with pytest.raises(ValueError, match='align'):
        p.replace_hidden(output, torch.ones(1, 2, 4))


def test_trace_all_position_restoration_and_cleanup(inventory, monkeypatch):
    model, targets, _ = inventory
    mods = p.attach(model, targets)
    with torch.no_grad():
        for m in mods.values():
            m.B.fill_(.01)
    monkeypatch.setattr(p.g, 'BOS', 0)
    monkeypatch.setattr(p.g, 'VOCAB', 256)
    result = p.trace_probe(model, [0, 2, 3], [0, 5, 7], 10, 'cpu')
    # Last block's full clean output gives exactly the clean distribution.
    assert result['restored'][-1] == pytest.approx(result['clean'])
    assert result['clean'] != result['corrupted']
    assert len(result['restored']) == len(model.layers)
    assert all(not b._forward_hooks for b in model.layers)
    assert all(m.enabled for m in mods.values())
    with pytest.raises(ValueError, match='lengths must align'):
        p.trace_probe(model, [0, 2, 3], [0, 5, 7, 8], 10, 'cpu')
    assert all(not b._forward_hooks for b in model.layers)


def test_trace_hooks_removed_after_model_error(inventory, monkeypatch):
    model, _, _ = inventory
    monkeypatch.setattr(p.g, 'BOS', 0)
    monkeypatch.setattr(p.g, 'logits_of', lambda *args: (_ for _ in ()).throw(RuntimeError('forward')))
    with pytest.raises(RuntimeError, match='forward'):
        p.trace_probe(model, [0, 2], [0, 3], 1, 'cpu')
    assert all(not b._forward_hooks for b in model.layers)


def test_checkpoint_roundtrip_and_tree_validation(inventory, tmp_path):
    model, targets, families = inventory
    mods = p.attach(model, targets)
    with torch.no_grad():
        for m in mods.values():
            m.B.fill_(.5)
    saved = dict(version=1, config={'rank': 16}, targets=[t.metadata() for t in targets],
                 families=families, state=p.a.trainable_state(mods))
    path = tmp_path / 'state.pt'
    torch.save(saved, path)
    new = Model()
    new_targets, new_families = p.enumerate_linears(new)
    _, chosen, loaded = p.load_state(new, new_targets, new_families, path)
    assert len(chosen) == len(targets)
    assert all(torch.equal(m.B, mods[n].B) for n, m in loaded.items())
    assert all(not par.requires_grad for par in new.parameters())
    saved['targets'][0]['layer'] = 99
    torch.save(saved, path)
    with pytest.raises(ValueError, match='mismatch'):
        p.load_state(Model(), new_targets, new_families, path)


def test_training_reuses_d2_sampling_smoothing_and_best_state(inventory, tmp_path, monkeypatch):
    model, targets, families = inventory
    corpus = build_corpus(rules='coarse')
    monkeypatch.setattr(p.g, 'BOS', 0)
    monkeypatch.setattr(p.g, 'DATA', tmp_path)
    monkeypatch.setattr(p.g, 'VOCAB', 256)
    (tmp_path / 'anchor-wikitext.json').write_text(json.dumps([1] * (150000 + 16 * 511)))
    monkeypatch.setattr(p.g, 'topk_records', lambda *args: [None] * 64)
    monkeypatch.setitem(p.g.__dict__, 'fsa', SimpleNamespace(nll_windows=lambda *args: [0.]))
    monkeypatch.setattr(p.a.c, 'checkpoint_recurrences', lambda *args: 0)
    calls = []
    def backward(student, batch, anchors, dev, vocab, arm, data):
        calls.append((len(batch), len(anchors), arm, data))
        sum(m.B.sum() for t in targets for m in t.linear.parametrizations.weight).backward()
        return {'loss': 0.}
    monkeypatch.setattr(p.a, 'training_backward', backward)
    accs = iter([.8, .8, .8, .1, .1])
    monkeypatch.setattr(p.a.evaluate, 'concept_acc', lambda *args: dict(acc=next(accs), n=len(args[-1])))
    args = p.parse_args(['--mode', 'train', '--run', str(tmp_path), '--out', str(tmp_path),
                         '--steps', '4', '--eval-every', '1', '--cases-per-policy', '2'])
    result = p.train(args, model, 'cpu', OffsetTokenizer(), corpus, targets, families, p.parameter_counts(targets))
    saved = torch.load(tmp_path / 'best_state.pt', weights_only=True)
    assert saved['step'] == 2 and saved['smoothed'] == pytest.approx(.8)
    assert calls == [(8, 2, 'placement', 'd2')] * 4
    assert saved['targets'] == [t.metadata() for t in targets]
    assert torch.allclose(next(iter(saved['state'].values()))['B'], torch.full((3, 16), -.002))
    assert all(r['dev']['n'] == 400 for r in result['trajectory'])
    assert result['trajectory'][0]['smoothed'] is None
    assert result['trajectory'][0]['dev']['se'] == pytest.approx(.02)


def test_cli_defaults():
    args = p.parse_args(['--mode', 'train', '--run', 'r', '--out', 'o'])
    assert (args.dev_large, args.targets, args.steps, args.eval_every) == (400, 'all', 150, 25)
    args = p.parse_args(['--mode', 'train', '--run', 'r', '--out', 'o', '--layers', '1,3'])
    assert args.layers == [1, 3]


@pytest.mark.parametrize('extra', [ ['--steps', '1'], ['--steps', '0'], ['--eval-every', '0'],
    ['--dev-large', '3'], ['--dev-large', '0'], ['--cases-per-policy', '3'],
    ['--layers', '-1'], ['--layers', '1,1'], ['--layers', 'oops'], ['--mode', 'trace'], ['--mode', 'attribute']])
def test_cli_invalid(extra):
    with pytest.raises(SystemExit):
        p.parse_args(['--mode', 'train', '--run', 'r', '--out', 'o'] + extra)


def test_trace_aggregates_first_option_token_and_baselines(inventory, monkeypatch):
    model, _, families = inventory
    corpus = build_corpus(rules='coarse')
    ids = [x['id'] for x in p.a.trained_policies(corpus, 1)]
    probes = p.a.large_dev_concepts(corpus, ids, 2, corpus['seed'])
    monkeypatch.setattr(p.g, 'BOS', 0)
    calls = []
    def traced(model, clean, corrupted, token, device):
        calls.append((clean, corrupted, token))
        return dict(clean=.8, corrupted=.2, restored=[.1 * i for i in range(6)])
    monkeypatch.setattr(p, 'trace_probe', traced)
    class CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=list(map(ord, text)))
    result = p.trace(model, CharTokenizer(), 'cpu', probes, corpus['policies'], families)
    assert result['n'] == 2 and result['clean'] == .8 and result['corrupted'] == .2
    assert all(token == ord(' ') and clean != corrupt for clean, corrupt, token in calls)
    assert result['blocks'][5]['restored_prob'] == .5
    assert result['rejected'] == 0
    assert all(len(clean) == len(corrupt) for clean, corrupt, _ in calls)
    assert '| 5 | none | True | 0.500000 |' in p.markdown(result)
    json.dumps(result, allow_nan=False)


def test_runner_reconstructs_checkpoint_dev_and_writes_reports(tmp_path, monkeypatch):
    corpus = build_corpus(rules='coarse')
    (tmp_path / 'corpus.json').write_text(json.dumps(corpus))
    model = Model()
    targets, families = p.enumerate_linears(model)
    chosen = [t for t in targets if t.layer == 1]
    mods = p.attach(model, chosen)
    ids = [x['id'] for x in p.a.trained_policies(corpus, 1)]
    checkpoint = tmp_path / 'best_state.pt'
    torch.save(dict(version=1, step=50, smoothed=.8,
                    config=dict(rank=16, trained_policy_ids=ids, dev_large=20),
                    targets=[t.metadata() for t in chosen], families=families,
                    state=p.a.trainable_state(mods)), checkpoint)
    monkeypatch.setattr(p.g, 'build', lambda: (Model(), 'cpu'))
    monkeypatch.setattr(p.g, 'load_tokenizer', lambda: OffsetTokenizer())
    expected = p.a.large_dev_concepts(corpus, ids, 20, corpus['seed'])
    def accuracy(model, tok, device, probes):
        assert probes == expected
        return dict(acc=.8, n=len(probes))
    monkeypatch.setattr(p.a.evaluate, 'concept_acc', accuracy)
    args = p.parse_args(['--mode', 'attribute', '--run', str(tmp_path), '--out', str(tmp_path / 'out'),
                         '--state', str(checkpoint)])
    assert args.dev_large == 400  # Checkpoint config, rather than this default, controls dev.
    result = p.run(args)
    assert result['baseline']['n'] == 20 and result['checkpoint_step'] == 50
    out = tmp_path / 'out'
    assert json.loads((out / 'result.json').read_text()) == result
    assert (out / 'inventory.json').is_file()
    assert (out / 'report.md').read_text() == p.markdown(result)


@pytest.fixture
def real_mesh(monkeypatch):
    # Use the deployed wrapper family, with several layers per BAR group.
    sys.path.insert(0, MESH_ROOT)
    from mesh.model.config import MeshModelConfig
    from mesh.model.hybrid import MeshModel
    monkeypatch.setenv('MESH_FROZEN_EXPERT_GROUPED', '0')
    monkeypatch.setenv('MESH_TRAINABLE_EXPERT_GROUPED', '0')
    torch.manual_seed(17)
    return MeshModel(MeshModelConfig(
        d_model=16, vocab_size=32, hybrid_override_pattern='WEWEWE',
        n_q_heads=2, n_kv_heads=1, head_dim=8, max_seq_len=32,
        n_routed_experts=2, num_experts_per_tok=2, n_shared_experts=1,
        moe_intermediate_size=24, use_block_attn_res=True,
        block_attn_res_block_size=3, use_mhc=False)).eval()


def test_real_bar_restores_complete_state_exactly(real_mesh, monkeypatch):
    monkeypatch.setattr(p.g, 'VOCAB', 32)
    model = real_mesh
    # Nonuniform BAR mixing catches stale completed-block statistics too.
    with torch.no_grad():
        for bar in [*model.block_attn_res, model.block_attn_res_head]:
            bar.proj.weight.normal_(std=.15)
    result = p.trace_probe(model, [0, 2, 3, 4], [0, 5, 7, 8], 10, 'cpu')
    assert result['clean'] != result['corrupted']
    assert result['restored_all'] == result['clean']
    assert result['restored_none'] == result['corrupted']
    assert result['restored'] == [result['clean']] * len(model.layers)
    # Compare against the untouched real model, not just our tracing forward.
    with torch.no_grad():
        expected = float(p.g.logits_of(model, torch.tensor([[0, 2, 3, 4]]))[0, -1].softmax(-1)[10])
    assert result['clean'] == expected
    assert '_forward_block_attn_res' not in model.__dict__


def test_real_bar_tracing_rejects_mhc_and_cleans_up(real_mesh):
    from dataclasses import replace
    real_mesh.config = replace(real_mesh.config, use_mhc=True)
    with pytest.raises(AssertionError, match='mHC off'):
        p.trace_probe(real_mesh, [0, 2], [0, 3], 1, 'cpu')
    real_mesh.config = replace(real_mesh.config, use_mhc=False)
    original = real_mesh.layers[0].forward
    def fail(*args, **kwargs):
        raise RuntimeError('forward')
    real_mesh.layers[0].forward = fail
    with pytest.raises(RuntimeError, match='forward'):
        p.trace_probe(real_mesh, [0, 2], [0, 3], 1, 'cpu')
    assert '_forward_block_attn_res' not in real_mesh.__dict__
    real_mesh.layers[0].forward = original


def test_whole_block_checkpoint_drops_effective_expert_weights_in_eval(real_mesh):
    targets, _ = p.enumerate_linears(real_mesh)
    modules = p.attach(real_mesh, targets)
    with torch.no_grad():
        for m in modules.values():
            m.B.normal_(std=.01)
    x = torch.tensor([[0, 2, 3, 4]])
    def forward_saved():
        shapes = []
        def pack(t):
            shapes.append(tuple(t.shape))
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            logits = p.g.logits_of(real_mesh, x)
        return logits, shapes
    before, saved = forward_saved()
    assert (16, 24) in saved or (24, 16) in saved
    before.square().sum().backward()
    grads = {n: m.B.grad.clone() for n, m in modules.items()}
    real_mesh.zero_grad(set_to_none=True)
    p.checkpoint_blocks(real_mesh)
    after, saved = forward_saved()
    assert (16, 24) not in saved and (24, 16) not in saved
    assert torch.equal(before, after)
    after.square().sum().backward()
    for n, m in modules.items():
        torch.testing.assert_close(m.B.grad, grads[n])
    assert not real_mesh.training


def test_real_msa_indexer_has_no_adapters_or_gradients(monkeypatch):
    sys.path.insert(0, MESH_ROOT)
    from mesh.model.attention import MSAAttentionLayer
    mixer = MSAAttentionLayer(16, 1, 1, kv_latent_dim=8, index_rank=8)
    model = nn.Module()
    model.layers = nn.ModuleList([mixer])
    targets, _ = p.enumerate_linears(model)
    index = [mixer.index_q_down, mixer.index_q_up, mixer.index_k_down, mixer.index_k_up]
    assert len(targets) == 5
    modules = p.attach(model, targets)
    assert all(not torch.nn.utils.parametrize.is_parametrized(m) for m in index)
    assert all(not m.weight.requires_grad for m in index)
    output = mixer(torch.randn(1, 4, 16))
    p.hidden(output).square().sum().backward()
    assert all(m.weight.grad is None for m in index)
    assert all(m.B.grad is not None for m in modules.values())
    assert 'fixed' in p.INDEXER_POLICY


def test_corruption_tries_alternatives_and_reports_rejections(inventory, monkeypatch):
    model, _, families = inventory
    corpus = build_corpus(rules='coarse')
    policy = p.a.trained_policies(corpus, 1)[0]
    probe = p.a.large_dev_concepts(corpus, [policy['id']], 2, corpus['seed'])[0]
    alternatives = []
    for other in corpus['policies']:
        if other['category'] == policy['category'] and other['attribute'] == policy['attribute'] and other['action'] != policy['action']:
            alternatives.extend(other['condition']['values'])
    assert len(alternatives) >= 2
    first, second = alternatives[:2]
    class LengthTokenizer:
        def encode(self, text, add_special_tokens=False):
            # First corruption shifts the sequence, the next aligns.
            case = dict(probe['case'], **{policy['attribute']: first})
            return SimpleNamespace(ids=[1] * (11 if p.a.render_ticket(case) in text else 10))
    corrupt, swap = p.corrupt_probe(probe, corpus['policies'], LengthTokenizer())
    assert swap['corrupted_value'] != first
    assert len(p.g.enc(LengthTokenizer(), corrupt['prompt'])) == len(p.g.enc(LengthTokenizer(), probe['prompt']))
    class RejectTokenizer:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1] * (10 if p.a.render_ticket(probe['case']) in text else 11))
    with pytest.raises(p.UnalignedCorruption):
        p.corrupt_probe(probe, corpus['policies'], RejectTokenizer())
    monkeypatch.setattr(p, 'trace_probe', lambda *args: pytest.fail('unaligned probe traced'))
    result = p.trace(model, RejectTokenizer(), 'cpu', [probe], corpus['policies'], families)
    assert result['n'] == 0 and result['rejected'] == 1
    assert result['clean'] is None
    json.dumps(result, allow_nan=False)
    assert 'rejected 1' in p.markdown(result)


def test_memory_probe_one_step_one_eval_largest_micro_and_exit(inventory, monkeypatch, tmp_path):
    model, targets, _ = inventory
    modules = p.attach(model, targets)
    opt = torch.optim.Adam([m.B for m in modules.values()])
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: calls.append('reset'))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 3 * 2**30)
    def backward(student, batch, anchors, *args):
        assert len(batch) == 2 and len(batch[0]['row']) == 80
        calls.append('backward')
        sum(m.B.sum() for m in modules.values()).backward()
    monkeypatch.setattr(p.a, 'training_backward', backward)
    monkeypatch.setitem(p.g.__dict__, 'fsa', SimpleNamespace(nll_windows=lambda *args: calls.append('nll')))
    monkeypatch.setattr(p, 'dev_accuracy', lambda *args: calls.append('dev'))
    result = p.probe_memory(model, None, 'cuda:0',
                            [dict(variants=[dict(row=[1] * 96), dict(row=[1] * 80)])],
                            [1, 2], None, [], opt)
    assert calls == ['reset', 'backward', 'reset', 'nll', 'dev']
    assert result['train_peak_gib'] == result['eval_peak_gib'] == 3
    assert opt.state  # The peak includes allocated Adam state.
    args = p.parse_args(['--mode', 'train', '--run', 'r', '--out', 'o', '--steps', '1', '--probe-memory'])
    assert args.probe_memory
    with pytest.raises(ValueError, match='CUDA'):
        p.probe_memory(model, None, 'cpu', [], [], None, [], opt)



def test_training_memory_probe_exits_before_selection_and_checkpoints(inventory, tmp_path, monkeypatch):
    model, targets, families = inventory
    corpus = build_corpus(rules='coarse')
    monkeypatch.setattr(p.g, 'BOS', 0)
    monkeypatch.setattr(p.g, 'DATA', tmp_path)
    (tmp_path / 'anchor-wikitext.json').write_text(json.dumps([1] * (150000 + 16 * 511)))
    monkeypatch.setattr(p.g, 'topk_records', lambda *args: [None] * 64)
    monkeypatch.setitem(p.g.__dict__, 'fsa', SimpleNamespace(nll_windows=lambda *args: pytest.fail('normal eval')))
    def probe(model, tok, device, acquisition, anchors, selection, probes, opt):
        assert acquisition and len(anchors) == 64 and len(probes) == 400
        assert all(block.forward.__name__ == 'wrapped' for block in model.layers)
        return dict(probe_memory=True, train_peak_gib=3., eval_peak_gib=4.)
    monkeypatch.setattr(p, 'probe_memory', probe)
    args = p.parse_args(['--mode', 'train', '--run', str(tmp_path), '--out', str(tmp_path),
                         '--probe-memory', '--steps', '1', '--cases-per-policy', '2'])
    result = p.train(args, model, 'cpu', OffsetTokenizer(), corpus, targets, families, p.parameter_counts(targets))
    assert result['probe_memory']
    assert not (tmp_path / 'log.jsonl').exists()
    assert not (tmp_path / 'best_state.pt').exists()
