"""S1 exact spans, streaming AUROC and value-only support regressions."""
import json

import numpy as np
import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, Tiny, cpu_harness
from test_c51 import WeightedModel
import census as C
import c54 as S


def test_exact_template_spans_and_recent_antecedent():
    facts = E.make_world(510051, n_people=3, per_person=2, relations=True)
    aliases = {0: ['TV']}
    items = S.labelled_items(TOK, BOS_T, facts, aliases)
    assert len(items) == len(facts)*(len(E.TRAIN_AUG_T)+3)
    assert {it['span_template'] for it in items} == {'train_aug', 'address_pronoun', 'address_pronoun_recent'}
    for item in items:
        assert len(item['span_labels']) == len(item['p'])+len(item['a'])-1
        seq = (item['p']+item['a'])[:-1]
        text = item['prompt']+' '+item['val']
        for span in item['spans']:
            assert item['span_labels'][span['token_start']:span['token_end']] == [span['pid']]*(span['token_end']-span['token_start'])
            assert TOK.decode(seq[span['token_start']:span['token_end']]) == text[span['start']:span['end']][:span['token_end']-span['token_start']]
        if item['span_template'] == 'address_pronoun_recent':
            pronouns = [s for s in item['spans'] if s['kind'] == 'pronoun']
            assert len(pronouns) == 1
            assert pronouns[0]['pid'] == item['mention_pids'][-1]
            assert {s['pid'] for s in item['spans'] if s['kind'] == 'name'} >= set(item['mention_pids'])
    relation = next(it for it in items if it['attr'] == "sister's name" and it['span_template'] == 'train_aug')
    assert any(s['pid'] != facts[relation['fi']].pid and s['token_start'] >= len(relation['p']) for s in relation['spans'])
    spans = S.mention_spans('TV saw Tamsin Vey; TVish did not.', facts, aliases)
    assert [(s['kind'], s['start'], s['end']) for s in spans] == [('name', 7, 17), ('alias', 0, 2)]


def test_offset_tokenization_boundary_merges_and_refuses_guessing():
    class OffsetTok:
        def encode(self, text, add_special_tokens=False):
            return type('Encoding', (), dict(ids=[7, 8], offsets=[(0, 3), (3, len(text))]))()
    labels, spans = S.token_labels(OffsetTok(), ' A B', [dict(start=1, end=2, pid=0, kind='name')], 1)
    assert labels.tolist() == [-2, 0, -2]
    assert spans[0]['token_start'] == 1 and spans[0]['token_end'] == 2
    with pytest.raises(ValueError, match='two different'):
        S.token_labels(OffsetTok(), ' A B', [dict(start=1,end=2,pid=0), dict(start=2,end=3,pid=1)])
    class NoOffsets:
        def encode(self, *args, **kwargs):
            return type('Encoding', (), dict(ids=[7]))()
    with pytest.raises(ValueError, match='requires tokenizer offsets'):
        S.token_labels(NoOffsets(), 'name', [])


def screen_fixture():
    m = Tiny(n=2, h=16, layers=2).eval()
    with torch.no_grad():
        m.emb.weight.zero_()
        m.emb.weight[:, 1] = 1
        m.emb.weight[10, 0], m.emb.weight[20, 0] = 1, -1
        for moe in m.moes:
            moe.router.weight.zero_()
            moe.router.weight[0, 0], moe.router.weight[1, 0] = 1, -1
            for expert in moe.experts:
                expert.up_proj.weight.zero_()
                expert.down_proj.weight.zero_()
            m.moes[0].experts[0].up_proj.weight[0, 0] = 2
            m.moes[0].experts[0].up_proj.weight[1, 1] = 1
    return m


def measure(m, batches, layers=None):
    screen = S.FiringScreen({0:'A', 1:'B'})
    census = C.Census(m, ['fwhq_sel'], min_routed=1, layers=layers, observer=screen)
    census.phase = 'screen'
    try:
        for tokens, labels in batches:
            screen.begin(labels)
            with torch.no_grad():
                m(torch.tensor(tokens).reshape(-1, 1))
        assert census.stream_tokens.tolist() == [0]
    finally:
        census.close()
    return screen


def test_planted_selectivity_unrouted_zeros_and_outputs(tmp_path):
    m = screen_fixture()
    screen = measure(m, [([10]*10, [0]*10), ([20]*10, [1]*10), ([20]*10, [-1]*10)], [0])
    summary = screen.output(tmp_path/'screen', {}, [], seed=7)
    assert summary['decision'] == 'value_only_control'
    top = summary['entities']['0']['top_units'][0]
    assert (top['layer'], top['expert'], top['unit']) == (0,0,0)
    assert top['auc_other'] == dict(estimate=1,lower=1,upper=1)
    assert top['auc_fineweb']['estimate'] == 1
    assert top['entity']['fired'] == 10
    assert top['entity']['mean_relu2'] == 4
    assert top['other']['routed'] == 0
    assert top['other']['firing_rate'] == 0
    assert summary['entities']['0']['counts']['other']['0.9']['certified'] == 2
    assert summary['entities']['1']['counts']['other']['0.9']['estimate'] == 0
    npz = np.load(tmp_path/'screen.npz')
    np.testing.assert_array_equal(npz['L0_E0_entity0_auc_other'][:,0], [1,1,1])
    assert npz['L0_E0_fineweb_hist_count'][0] == 10
    assert json.loads((tmp_path/'screen.summary.json').read_text()) == summary
    restriction, payload = S.load_unit_restriction(m, summary['entities']['0']['top8_file'], [0])
    assert len(payload['units']['moes.0.experts.0']) == 8
    base_up = m.moes[0].experts[0].up_proj.weight.detach().clone()
    E.attach(m, [0], 'fsa_full', 0, projs=('down_proj',), restrict=restriction)
    for ei, expert in enumerate(m.moes[0].experts):
        param = expert.down_proj.parametrizations.weight[0]
        allowed = restriction.get(f'moes.0.experts.{ei}.down_proj', ('cols',torch.zeros(16,dtype=torch.bool)))[1]
        assert torch.equal(param.mask.any(0), allowed)
        assert not hasattr(expert.up_proj, 'parametrizations')
    assert torch.equal(m.moes[0].experts[0].up_proj.weight, base_up)
    assert not m.moes[0].router._forward_hooks


def test_streaming_chunking_duplicates_and_padding(tmp_path):
    m = WeightedModel()
    screen = measure(m, [([10]*3, [0,1,-1])])
    for key in ('L0_E0', 'L0_E1'):
        acc = screen.units[key]
        assert acc['routed'] == {0:1, 1:1, -1:1}
        assert acc['relu2_sum'][0][0] == 4
    result = screen.output(tmp_path/'ties', {}, [])
    assert result['entities']['0']['top_units'][0]['auc_other']['estimate'] == .5
    assert result['decision'] == 'train_addresses'
    m = screen_fixture()
    a = measure(m, [([10,20,20,10], [0,1,-1,0])])
    b = measure(m, [([10,20], [0,1]), ([20,10], [-1,0])])
    for key, acc in a.units.items():
        for group in a.groups:
            for left,right in zip(a.histogram(acc,[group]), b.histogram(b.units[key],[group])):
                np.testing.assert_array_equal(left,right)


def test_histogram_exact_bf16_auc_and_fp32_uncertainty():
    rng = np.random.default_rng(11)
    p = torch.tensor(rng.normal(size=(17,4)),dtype=torch.bfloat16)
    n = torch.tensor(rng.normal(size=(23,4)),dtype=torch.bfloat16)
    screen = S.FiringScreen({0:'A',1:'B'})
    for group,z in [(0,p),(1,n)]:
        screen.begin([group]*len(z))
        screen.observe('x',dict(layer=0,expert=0,name='x'),z,torch.arange(len(z)))
    acc = screen.units['x']
    auc = S.auc_hist(screen.histogram(acc,[0]),screen.histogram(acc,[1]),17,23,4)
    pp,nn = p.float().clamp_min(0).numpy(),n.float().clamp_min(0).numpy()
    exact = ((pp[:,None,:] > nn[None,:,:]) + .5*(pp[:,None,:] == nn[None,:,:])).mean((0,1))
    np.testing.assert_allclose(auc, np.tile(exact,(3,1)), atol=1e-15)
    screen = S.FiringScreen({0:'A',1:'B'})
    for group,z in [(0,1.001),(1,1.002)]:
        screen.begin([group])
        screen.observe('x',dict(layer=0,expert=0,name='x'),torch.tensor([[z]]),torch.tensor([0]))
    auc = S.auc_hist(screen.histogram(screen.units['x'],[0]),screen.histogram(screen.units['x'],[1]),1,1,1)
    np.testing.assert_array_equal(auc[:,0], [.5,0,1])


@pytest.mark.parametrize('units', [{'bad':[0]}, {'moes.0.experts.0':[16]}, {'moes.0.experts.0':[0,0]}, {'moes.0.experts.0':[True]}, {}])
def test_invalid_unit_lists_fail(tmp_path,units):
    path = tmp_path/'units.json'
    path.write_text(json.dumps(dict(version=1,units=units)))
    with pytest.raises(ValueError):
        S.load_unit_restriction(screen_fixture(),path,[0])


def test_opt_in_parser_and_guard(tmp_path):
    parsed = E.build_parser().parse_args(['--method','fsa_full','--out','unused','--down-only','--restrict-units','list.json'])
    assert str(parsed.restrict_units) == 'list.json'
    with pytest.raises(SystemExit,match='value-only'):
        E.main(['--method','fsa_full','--out','unused','--restrict-units','list.json'])
    with pytest.raises(ValueError,match='requires stream fwhq_sel'):
        C.main(['--firing-screen','--stream','web='+str(tmp_path/'missing.json'),'--anchor','unused','--out','unused'])


def test_real_screen_main_dev_and_arm_identity(cpu_harness, tmp_path):
    stream = tmp_path/'fwhq.u32'
    np.full(64,10,dtype='<u4').tofile(stream)
    ref = tmp_path/'reference.json'
    ref.write_text(json.dumps([99]*64))
    tokenizer = tmp_path/'tokenizer.json'
    tokenizer.write_text('{}')
    export = tmp_path/'export'
    export.mkdir()
    (export/'weights').write_bytes(b'fixture')
    flags = ['--firing-screen','--screen-layers','0,1','--stream',f'fwhq_sel={stream}',
             '--tokens','64','--context-length','32','--batch-size','2',
             '--factqa-people','3','--factqa-facts','2','--factqa-relations',
             '--export',str(export),'--anchor',str(ref),'--eval-panels',str(ref),
             '--tokenizer',str(tokenizer)]
    dev = C.main([*flags,'--out',str(tmp_path/'dev.npz')])
    assert dev['fineweb_tokens'] == 64
    assert dev['counts']['-1'] == 64
    assert dev['layers'] == [0,1]
    assert len(dev['entities']) == 3
    assert all(f['name'].startswith('CensusDev510051') for f in dev['facts'])
    names = {f['name'] for f in dev['facts']}
    assert all(f['val'] in names for f in dev['facts'] if f['attr'] == "sister's name")
    arm = C.main([*flags,'--factqa-world','arm','--out',str(tmp_path/'arm.npz')])
    assert arm['facts'] == [f._asdict() for f in E.make_world(510051,3,2,relations=True)]
    assert (tmp_path/'arm.spans.json').exists()
    assert str(export.resolve()/'weights') in arm['input_sha256']
    assert str(tokenizer.resolve()) in arm['input_sha256']


def test_real_value_only_write_and_content_resume_guard(cpu_harness, tmp_path):
    run, _ = cpu_harness
    path = tmp_path/'units.json'
    path.write_text(json.dumps(dict(version=1,units={'moes.0.experts.0':[1,4,7,8]})))
    result, patch = run(['--down-only','--restrict-units',str(path),'--steps','2','--sparse-state','--budget','0','--bop-tau-abs','1e-12',
                         '--ckpt-every','1'], 'restricted')
    state = torch.load(patch,map_location='cpu',weights_only=False)
    assert state['meta']['unit_restriction']['units'] == {'moes.0.experts.0':[1,4,7,8]}
    for name, entry in state['entries'].items():
        assert name.endswith('.down_proj')
        slots = entry['idx'].numpy()
        columns = slots % entry['shape'][1]
        if name == 'moes.0.experts.0.down_proj':
            assert set(columns) <= {1,4,7,8}
            assert len(slots) > 0
        else:
            assert not len(slots)
    path.write_text(json.dumps(dict(version=1,units={'moes.0.experts.0':[2,4,7,8]})))
    with pytest.raises(SystemExit,match='config mismatch'):
        run(['--down-only','--restrict-units',str(path),'--steps','2','--sparse-state','--budget','0','--bop-tau-abs','1e-12',
             '--ckpt-every','1','--resume','auto'],'restricted')


def test_threshold_boundary_and_separate_fineweb_comparison(tmp_path):
    screen = S.FiringScreen({0:'A',1:'B'})
    for group,z in [(0,[1.]*80+[0.]*20),(1,[0.]*100),(-1,[1.]*100)]:
        screen.begin([group]*len(z))
        screen.observe('x',dict(layer=12,expert=0,name='moes.12.experts.0'),
                       torch.tensor(z)[:,None],torch.arange(len(z)))
    summary = screen.output(tmp_path/'boundary',{},[])
    entity = summary['entities']['0']
    assert entity['top_units'][0]['auc_other'] == dict(estimate=.9,lower=.9,upper=.9)
    assert entity['top_units'][0]['auc_fineweb']['estimate'] == .4
    assert entity['counts']['other']['0.9'] == dict(estimate=1,certified=1,unresolved=0)
    assert entity['counts']['fineweb']['0.8']['certified'] == 0
    assert summary['decision'] == 'value_only_control'
