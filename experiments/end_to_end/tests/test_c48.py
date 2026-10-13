"""C48: checkpoint payload/inventory identity with unchanged manifests."""
import pytest

from test_c46 import args
import c44_runtime as R


@pytest.mark.parametrize('source', ['export', 'bench-dir'])
@pytest.mark.parametrize('mutation', ['replace', 'add', 'delete'])
def test_resume_refuses_directory_payload_changes(tmp_path, source, mutation):
    directory = tmp_path/'inputs'
    (directory/'relay_pack').mkdir(parents=True)
    manifest = directory/'manifest.json'
    manifest.write_text('{"packed_experts": {"expert0": "relay_pack/weights.pt"}}')
    payload = directory/'relay_pack/weights.pt'
    payload.write_bytes(b'original weights')
    panel = directory/'arc_easy_200.json'
    panel.write_text('[{"question": "original"}]')
    flags = ['--'+source, str(directory)]
    checkpoint = R.Checkpoints(args(tmp_path, *flags))
    checkpoint.save(1)
    assert checkpoint.inputs[str(payload)] == R.hashlib.sha256(payload.read_bytes()).hexdigest()
    assert str(panel) in checkpoint.inputs
    target = payload if source == 'export' else panel
    if mutation == 'replace':
        target.write_bytes(b'different contents')
    elif mutation == 'add':
        (directory/'new_shard.json').write_text('{}')
    else:
        target.unlink()
    assert manifest.read_text() == '{"packed_experts": {"expert0": "relay_pack/weights.pt"}}'
    with pytest.raises(SystemExit, match='input contents changed'):
        R.Checkpoints(args(tmp_path, *flags, '--resume', 'auto'))


def test_resume_accepts_identical_directory_contents_and_path_spelling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'export').mkdir()
    (tmp_path/'export/manifest.json').write_text('{}')
    (tmp_path/'export/shard.pt').write_bytes(b'weights')
    (tmp_path/'bench').mkdir()
    (tmp_path/'bench/panel.json').write_text('[]')
    R.Checkpoints(args(tmp_path, '--export', './export', '--bench-dir', './bench')).save(1)
    resumed = R.Checkpoints(args(tmp_path, '--export', str(tmp_path/'export'),
                                '--bench-dir', str(tmp_path/'bench'), '--resume', 'auto'))
    assert resumed.saved['step'] == 1
