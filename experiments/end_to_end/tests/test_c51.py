"""C51 energy, actual weighted dispatch, held-out ranks and routing diagnostics."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from test_e2e_fsa import E, TOK, BOS_T, Tiny
from test_c49 import planted
from test_fast_moe import FakeMoE
import census as C


class WeightedMoE(FakeMoE):
    def forward(self, x):
        n = x.numel()//x.shape[-1]
        # Duplicate expert 0 must be coalesced to weight .75, not .25²+.5².
        ids = torch.tensor([[0,0,1]], device=x.device).expand(n,-1)
        weights = torch.tensor([[.25,.5,.25]], device=x.device).expand(n,-1)
        return self._forward_impl(x, (ids,weights))


class WeightedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128,4)
        self.moe = WeightedMoE(d=4,n=2,h=20,top_k=3)
        self.moe.fp8_pad = True
        with torch.no_grad():
            self.emb.weight.fill_(1)
            for e in self.moe.experts:
                e.up_proj.weight.fill_(.5)  # z=2, phi=4, phi²=16
                e.down_proj.weight.fill_(.25)  # ||v||²=.25

    def forward(self, tokens):
        return self.moe(self.emb(tokens))


def test_fourth_moment_changes_rare_burst_ranking_and_preserves_raw_fields(tmp_path):
    m = planted()
    # Running-mean fixture: one-token windows give an exact rare-burst example.
    with torch.no_grad():
        m.emb.weight[10,0] = 0
        m.emb.weight[20,0] = 1
        e = m.moes[0].experts[0]
        e.up_proj.weight[3].zero_()
        e.up_proj.weight[3,0] = 10
        e.up_proj.weight[4].zero_()
        e.up_proj.weight[4,1] = 2
    stream = {'web':np.array([10]*99+[20]+[10]*100)}
    stats = C.Census(m,stream,min_routed=1,split_fraction=.5)
    try:
        C.run(m,stream,1,[200],7,'cpu',stats)
        a = stats.derived('L0_E0','A')
        assert a['mean_relu2'][0,3] == 1 < a['mean_relu2'][0,4]
        assert a['mean_relu4'][0,3] == 100 > a['mean_relu4'][0,4]
        assert a['energy'][0,3] == 25
        assert a['energy'][0,4] == 4
        rank = stats.ranks()[0].tolist()
        assert rank.index(4) < rank.index(3)
        stats.output(tmp_path/'energy',{},plots=False)
        out = np.load(tmp_path/'energy.npz')
        assert out['L0_E0_relu2_sum'][0,3] == 100
        assert out['L0_E0_relu4_sum'][0,3] == 10000
        assert out['L0_E0_legacy_energy'][0,3] == .25
    finally:
        stats.close()


def test_actual_topk_weights_duplicates_padding_and_cleanup(tmp_path):
    m = WeightedModel().eval()
    before = m(torch.tensor([[10,10,10]])).detach()
    stats = C.Census(m,['web'],min_routed=1)
    try:
        C.run(m,{'web':np.full(9,10)},3,[9],2,'cpu',stats)
        for key, energy in [('L0_E0',2.25),('L0_E1',.25)]:
            a = stats.derived(key)
            assert stats.units[key]['routed'].item() == 9
            assert stats.units[key]['relu4_sum'][0,0].item() == 9*16
            assert a['energy'][0,0] == 4
            assert a['unconditional_energy'][0,0] == energy
        assert stats.dispatch[0]['top_k'] == 3
        stats.output(tmp_path/'weighted',{},plots=False)
    finally:
        stats.close()
    assert '_forward_frozen_exact_grouped' not in m.moe.__dict__
    assert not m.moe.experts[0].up_proj._forward_hooks
    assert torch.equal(m(torch.tensor([[10,10,10]])).detach(),before)
    assert all(p.grad is None for p in m.parameters())


def test_routing_probability_is_in_unconditional_denominator():
    m = Tiny(n=2,h=20,layers=1).eval()
    with torch.no_grad():
        m.emb.weight.zero_()
        m.emb.weight[:,1] = 1
        m.emb.weight[10,0], m.emb.weight[20,0] = 1,-1
        r = m.moes[0].router.weight
        r.zero_(); r[0,0], r[1,0] = 1,-1
        for e in m.moes[0].experts:
            e.up_proj.weight.zero_(); e.up_proj.weight[:,1] = 2
            e.down_proj.weight.fill_(.125)
    stats = C.Census(m,['web'],min_routed=1)
    try:
        C.run(m,{'web':np.array([10]*3+[20]+[10]*3+[20])},1,[8],3,'cpu',stats)
        a,b = stats.derived('L0_E0'),stats.derived('L0_E1')
        assert a['energy'][0,0] == b['energy'][0,0] == 4
        assert a['unconditional_energy'][0,0] == 3
        assert b['unconditional_energy'][0,0] == 1
        assert a['unconditional_frequency'][0,0] == .75
        assert b['unconditional_frequency'][0,0] == .25
    finally:
        stats.close()


def test_tau_calibration_frozen_on_A_and_strict_tail_frequency(tmp_path,monkeypatch):
    m = planted()
    with torch.no_grad():
        m.moes[0].experts[0].up_proj.weight[3,1] = 3
    stream = {'web':np.full(128,10)}
    stats = C.Census(m,stream,min_routed=1,split_fraction=.5)
    monkeypatch.setattr(C,'TAU_TOKENS',64)
    try:
        assert C.calibrate(m,stream,32,[128],2,'cpu',stats) == [64]
        assert stats.stream_tokens.tolist() == [0]
        assert stats.units['L0_E0']['routed'].tolist() == [0]
        assert 1 <= stats.tau[0] < 1.023
        C.run(m,stream,32,[128],2,'cpu',stats)
        a = stats.derived('L0_E0')
        assert a['tail_frequency'][0,3] == 1
        assert a['tail_frequency'][0,4] == 0
        assert stats.units['L0_E0']['tail'][0,3] == 128
        summary = stats.output(tmp_path/'tail',{},False)
        assert summary['tau']['0'] == stats.tau[0]
    finally:
        stats.close()


def test_unobserved_excluded_from_all_ranks_even_when_B_has_evidence(tmp_path):
    m = planted()
    stream = {'web':np.full(96,10)}
    stats = C.Census(m,stream,min_routed=64,split_fraction=.5)
    try:
        C.run(m,stream,32,[96],2,'cpu',stats)
        assert stats.split_units['L0_E0']['A']['routed'].tolist() == [64]
        assert stats.split_units['L0_E0']['B']['routed'].tolist() == [32]
        out = stats.output(tmp_path/'unknown',{},False)
        assert out['layers']['0']['ranked_A'] == 512
        assert out['layers']['0']['ranked_B'] == 0
        assert out['layers']['0']['unobserved_cells_B'] == 1
        assert out['layers']['0']['rank_stability']['10']['jaccard'] == 0
        data = np.load(tmp_path/'unknown.npz')
        assert np.isnan(data['L0_E0_energy_B']).all()
    finally:
        stats.close()
    stats = C.Census(m,stream,min_routed=97)
    try:
        C.run(m,stream,32,[96],1,'cpu',stats)
        summary = stats.output(tmp_path/'none',{},False)['layers']['0']
        assert summary['unobserved_cells'] == 1
        assert summary['near_dormant_everywhere'] == summary['domain_specific'] == 0
        assert len(np.load(tmp_path/'none.npz')['layer0_rank']) == 0
    finally:
        stats.close()


def test_selection_A_and_jaccard_B_are_separate(tmp_path):
    m = planted()
    with torch.no_grad():
        e = m.moes[0].experts[0]
        e.up_proj.weight[3:103].zero_()
        e.up_proj.weight[3:53,0] = 1
        e.up_proj.weight[53:103,0] = -1
    stream = {'web':np.array([10]*64+[20]*64)}
    stats = C.Census(m,stream,min_routed=1,split_fraction=.5)
    try:
        C.run(m,stream,32,[128],3,'cpu',stats)
        out = stats.output(tmp_path/'split',{},False)
        data = np.load(tmp_path/'split.npz')
        a,b = data['layer0_rank'],data['layer0_rank_B']
        assert 3 in C.bottom(a,.1) and 3 not in C.bottom(b,.1)
        expected = len(C.bottom(a,.1)&C.bottom(b,.1))/len(C.bottom(a,.1)|C.bottom(b,.1))
        assert out['layers']['0']['rank_stability']['10']['jaccard'] == expected < 1
        np.testing.assert_array_equal(a,stats.ranks('A')[0])
        assert stats.stream_tokens.tolist() == [128]
        assert stats.split_tokens['A'].tolist() == stats.split_tokens['B'].tolist() == [64]
    finally:
        stats.close()


def test_batch1_agreement_subsample_does_not_mutate_main_census(monkeypatch):
    m = planted()
    stream = {'web':np.full(128,10)}
    stats = C.Census(m,stream,min_routed=1,split_fraction=.5)
    try:
        C.run(m,stream,32,[128],3,'cpu',stats)
        before = copy.deepcopy(stats.state())
        monkeypatch.setattr(C,'AGREEMENT_TOKENS',96)
        result = C.batch_agreement(m,stream,32,[128],3,'cpu',stats)
        assert result['processed_tokens'] == 96
        assert result['batch_sizes'] == [1,3]
        assert result['layers']['0']['membership_overlap'] == 1
        assert result['passed'] is True and result['pass_threshold'] == .95
        for field in C.FIELDS:
            assert torch.equal(before['units']['L0_E0'][field],stats.units['L0_E0'][field])
        assert stats.active and stats.stream_tokens.tolist() == [128]
    finally:
        stats.close()


def test_factqa_training_only_dev_names_and_answer_reachability():
    facts,items = C.factqa_items(TOK,BOS_T,people=2,facts_per_person=2)
    assert (facts,items) == C.factqa_items(TOK,BOS_T,people=2,facts_per_person=2)
    assert len(items) == 4*len(E.TRAIN_AUG_T)
    test_names = {f.name for f in E.make_world(0,n_people=3,per_person=40)}
    assert {f.name for f in facts}.isdisjoint(test_names)
    assert all(f.name.startswith('CensusDev510051') for f in facts)
    assert {it['ti'] for it in items} == set(range(len(E.TRAIN_AUG_T)))
    m = planted()
    stats = C.Census(m,['web'],min_routed=1)
    try:
        C.run(m,{'web':np.full(128,10)},32,[128],2,'cpu',stats)
        before = stats.units['L0_E0']['routed'].clone()
        result = C.factqa_probe(m,items,'cpu',stats)
        layer = result['layers']['0']
        assert result['domain'] == 'factqa' and result['prompts'] == len(items)
        assert layer['routing_share'] == layer['routing_probability'] == [1]
        assert layer['bottom10_dispatched_at_answer'] == [51]*len(items)
        assert layer['tokens'] == sum(len(it['p'])+len(it['a'])-1 for it in items)
        assert torch.equal(before,stats.units['L0_E0']['routed'])
    finally:
        stats.close()


def test_batch_agreement_reports_failure_for_batch_sensitive_units():
    class BatchSensitive(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = planted()
            with torch.no_grad():
                e = self.model.moes[0].experts[0]
                e.up_proj.weight[3:243].zero_()
                e.up_proj.weight[3:123,0] = 1
                e.up_proj.weight[123:243,0] = -1

        def forward(self, tokens):
            return self.model(torch.full_like(tokens,20) if len(tokens) > 1 else tokens)

    m = BatchSensitive().eval()
    streams = {'web':np.full(128,10)}
    stats = C.Census(m,streams,min_routed=1)
    try:
        result = C.batch_agreement(m,streams,32,[128],2,'cpu',stats)
        assert result['layers']['0']['membership_overlap'] < .95
        assert result['passed'] is False
        assert stats.stream_tokens.tolist() == [0]
    finally:
        stats.close()


def test_unsupported_dispatch_fails_and_restores_hooks():
    m = planted()
    m.moes[0].top_k = 2
    with pytest.raises(ValueError,match='grouped dispatch'):
        C.Census(m,['web'])
    assert not m.moes[0].router._forward_hooks


def test_checkpoint_restores_new_statistics_and_rejects_legacy(tmp_path):
    m = planted()
    stream = {'web':np.full(160,10)}
    full = C.Census(m,stream,min_routed=1)
    try:
        C.calibrate(m,stream,32,[160],2,'cpu',full)
        C.run(m,stream,32,[160],2,'cpu',full)
        expected = copy.deepcopy(full.state())
    finally:
        full.close()
    first = C.Census(m,stream,min_routed=1)
    try:
        C.calibrate(m,stream,32,[160],2,'cpu',first)
        cursors = C.run(m,stream,32,[160],2,'cpu',first,stop_after=64)
        state = copy.deepcopy(first.state())
    finally:
        first.close()
    restored = C.Census(m,stream,min_routed=1)
    try:
        with pytest.raises(ValueError,match='C51 checkpoints'):
            restored.restore(state['units'])
        restored.restore(state)
        C.run(m,stream,32,[160],2,'cpu',restored,cursors=cursors)
        for key in expected['units']:
            for field in C.FIELDS:
                assert torch.equal(expected['units'][key][field],restored.units[key][field])
                for split in ('A','B'):
                    assert torch.equal(expected['split_units'][key][split][field],restored.split_units[key][split][field])
        assert restored.tau == expected['tau']
        assert torch.equal(restored.stream_tokens,expected['stream_tokens'])
    finally:
        restored.close()


@pytest.mark.parametrize('flag,value',[('--min-routed','0'),('--split-fraction','0'),
    ('--split-fraction','1'),('--split-fraction','nan'),('--factqa-seed','1')])
def test_main_rejects_invalid_new_configuration(flag,value):
    with pytest.raises(ValueError,match='invalid census'):
        C.main(['--stream','web=unused','--anchor','unused','--out','unused',flag,value])


def test_c56_defaults_and_70_30_window_boundary():
    a = C.build_parser().parse_args(['--stream','web=unused','--anchor','unused','--out','unused'])
    assert a.min_routed == 10000 and a.split_fraction == .3
    stats = C.Census(planted(), ['web'])
    try:
        assert stats.min_routed == 10000 and stats.split_fraction == .3
        assert C.split_boundary(100*32,32,stats.split_fraction) == 70*32
    finally:
        stats.close()


def test_tau_uses_nonzero_median_not_zero_inclusive_p99():
    stats = C.Census(planted(), ['web'])
    try:
        hist = stats.histograms[0]
        hist[0] = 1000000
        # phi .25 and 4; zeros dominate but only positive observations set tau.
        hist[961], hist[1089] = 100, 25
        stats.freeze_tau()
        assert .25 <= stats.tau[0] < .256
        assert stats.phase == 'measure' and stats.stream_tokens.sum() == 0
        hist[961], hist[1089] = 100, 100
        stats.freeze_tau()
        assert 2.125 <= stats.tau[0] < 2.172
        hist.zero_(); hist[0] = 100
        with pytest.raises(ValueError,match='nonzero phi'):
            stats.freeze_tau()
        hist.zero_(); hist[-1] = 100
        with pytest.raises(ValueError,match='median exceeds'):
            stats.freeze_tau()
    finally:
        stats.close()


def test_per_expert_selection_and_confirmation_do_not_pool_energy(tmp_path):
    stats = C.Census(WeightedModel(), ['web'], min_routed=1)
    try:
        stats.stream_tokens[:] = 200
        for split in ('A','B'):
            stats.split_tokens[split][:] = 100
            for ei in (0,1):
                acc = stats.split_units[f'L0_E{ei}'][split]
                acc['routed'][:] = 100
                energies = torch.arange(1,21,dtype=torch.float64) + ei*1000
                if split == 'B' and ei == 1:
                    energies = energies.flip(0)
                acc['weighted_relu4_sum'][0] = energies
        ranks = stats.expert_ranks()
        assert C.expert_bottom(ranks[0],.2) == {0,1,2,3,20,21,22,23}
        assert C.bottom(stats.ranks()[0],.2) == set(range(8))
        summary = stats.output(tmp_path/'experts',{},False)['layers']['0']
        assert summary['per_expert']['0']['rank_stability']['20']['jaccard'] == 1
        assert summary['per_expert']['1']['rank_stability']['20']['jaccard'] == 0
        assert summary['rank_stability']['20']['jaccard'] == pytest.approx(1/3)
        data = np.load(tmp_path/'experts.npz')
        np.testing.assert_array_equal(data['layer0_E1_rank'], np.arange(20,40))
        state = copy.deepcopy(stats.state()); state.pop('policy')
        with pytest.raises(ValueError,match='selection policy'):
            stats.restore(state)
    finally:
        stats.close()


def test_bottom20_batch_check_fails_even_when_pooled_overlap_passes():
    class UnequalExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = WeightedModel()
            e0,e1 = self.model.moe.experts
            e0.up_proj = nn.Linear(4,1000,bias=False)
            e0.down_proj = nn.Linear(1000,4,bias=False)
            with torch.no_grad():
                e0.up_proj.weight.fill_(.5); e0.down_proj.weight.fill_(.25)
                e1.up_proj.weight.copy_(torch.arange(1,21)[:,None].expand(-1,4))
            self.batched = False
            e1.up_proj.register_forward_hook(lambda mod,inp,out: out.flip(-1) if self.batched else out)

        def forward(self, tokens):
            self.batched = len(tokens) > 1
            return self.model(tokens)
    model = UnequalExperts().eval()
    streams = {'web':np.full(128,10)}
    stats = C.Census(model,streams,min_routed=1)
    try:
        result = C.batch_agreement(model,streams,32,[128],2,'cpu',stats)
        layer = result['layers']['0']
        assert result['bottom_fraction'] == .2 and result['selection'] == 'per expert'
        assert layer['membership_overlap'] > .95
        assert layer['per_expert']['0']['passed'] is True
        assert layer['per_expert']['1']['passed'] is False
        assert result['passed'] is False
    finally:
        stats.close()
