from types import SimpleNamespace

import pytest

from visual_rl import train


@pytest.mark.parametrize('degree,model_name,allowed', [(2, 'minimax_h3', True), (1, 'minimax_h3', False),
                                                       (2, 'other_backend', False)])
def test_actual_cli_counts_logical_h3_workers_and_launches_all_physical_workers(monkeypatch, degree, model_name, allowed):
    events = []
    config = SimpleNamespace(model=SimpleNamespace(name=model_name, options={
        'context_parallel_size': degree, 'finetune': {'mode': 'full'}}),
        runtime=SimpleNamespace(distributed='fsdp', log='jsonl'), algorithm=SimpleNamespace(name='diffusionart'),
        sampling=SimpleNamespace(prompts_per_rollout=32))
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.setattr(train, 'load_config', lambda *args: config)
    monkeypatch.setattr(train, 'get_backend', lambda config: SimpleNamespace())
    monkeypatch.setattr('visual_rl.prepare_assets.plan_environment', lambda path: ([], []))
    monkeypatch.setattr('visual_rl.prepare_assets.validate_bindings', lambda *args: None)
    monkeypatch.setattr('visual_rl.prepare_distributed.prepare_shared_assets', lambda *args, **kwargs: events.append('prepare'))
    def execute(executable, command):
        assert '--nnodes=8' in command and '--nproc-per-node=8' in command
        events.append('launch64')
        raise SystemExit(0)
    monkeypatch.setattr(train.os, 'execv', execute)
    command = ['--config', 'unused.yaml', '--nproc-per-node', '8', '--nnodes', '8', '--master-addr', '127.0.0.1']
    if allowed:
        with pytest.raises(SystemExit) as result:
            train.main(command)
        assert result.value.code == 0 and events == ['prepare', 'launch64']
    else:
        assert train.main(command) == 2 and events == []
