"""Offline deployment boundaries; never invokes SSH, systemd or devices."""
from pathlib import Path
import json
from unittest.mock import patch
import pytest
from tools.deployment import (PHYSICAL, Deployment, config_differences,
    require_config_compatible, require_safe_states, render_unit, safe_release,native_profile_assets)


@pytest.mark.parametrize('name',['../x','a/b','a b',';reboot','', '/tmp/x'])
def test_release_path_rejects_escape(tmp_path,name):
    with pytest.raises(ValueError): safe_release(tmp_path,name)


def test_release_path_is_explicit_child(tmp_path):
    assert safe_release(tmp_path,'deploy-76')==tmp_path/'IDCS-releases/deploy-76'


def test_templates_replace_source_not_operational_options():
    text='[Service]\nWorkingDirectory=@RELEASE@\nEnvironment=PYTHONPATH=@RELEASE@\nExecStart=/bin/python -m pc.streamer --verified-rtp-headers\n'
    rendered=render_unit(text,Path('/releases/one'))
    assert f'WorkingDirectory={Path("/releases/one")}' in rendered
    assert '--verified-rtp-headers' in rendered


@pytest.mark.parametrize('text',['[Service]\nExecStart=x\n','[Service]\nWorkingDirectory=/x/IDCS-runtime\n','[Service]\nWorkingDirectory=/x/IDCS-trials/old\n'])
def test_templates_refuse_ambiguous_or_old_source(text):
    with pytest.raises(ValueError): render_unit(text,Path('/releases/one'))


def test_readonly_hud_additions_are_explicit_not_arbitrary():
    old={'operator':{'lock_lost_s':1},'rpi':{'operator_display':{'sink':'wayland'}}}
    new={'operator':{'lock_lost_s':1,'hud':{'push_endpoint':'tcp://pi:5594'}},
         'rpi':{'operator_display':{'sink':'wayland','hud_sim_bind':'tcp://*:5594'}}}
    assert len(require_config_compatible(old,new))==2
    with pytest.raises(ValueError): require_config_compatible(old,{'operator':{'lock_lost_s':2}})
    with pytest.raises(ValueError): require_config_compatible(old,{'operator':{'lock_lost_s':1,'motor_enable':True}})


@pytest.mark.parametrize('name',sorted(PHYSICAL))
def test_any_physical_owner_blocks_deployment(name):
    with pytest.raises(RuntimeError): require_safe_states({name:{'ActiveState':'active'}})
    require_safe_states({name:{'ActiveState':'inactive'}})


def test_unchanged_config_and_removed_key():
    assert config_differences({'x':1},{'x':1})==[]
    with pytest.raises(ValueError): require_config_compatible({'x':1},{})


def test_restart_refuses_motor_even_if_previously_active(tmp_path):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    with patch.object(manager,'run') as run:
        with pytest.raises(RuntimeError): manager.restart_previous({'idcs-controller.service':{'ActiveState':'active'}})
        run.assert_not_called()


def test_software_session_recovery_is_not_hil(tmp_path):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    with patch.object(manager,'run') as run,patch.object(manager,'state',return_value={'ActiveState':'active'}):
        manager.restart_previous({'idcs-sim.target':{'ActiveState':'active'},'idcs-sim-ui.service':{'ActiveState':'active'}})
        run.assert_called_once_with(['systemctl','--user','start','idcs-sim.target'])


def test_install_cannot_write_outside_unit_directories(tmp_path):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    with pytest.raises(ValueError): manager.install(tmp_path/'other.service','x')


def test_jetson_system_operations_require_noninteractive_sudo(tmp_path):
    manager=Deployment(tmp_path,'jetson','one',templates=tmp_path)
    assert manager.ctl==['sudo','-n','systemctl']


def test_missing_native_parser_blocks_before_any_activation(tmp_path):
    (tmp_path/'engine').write_bytes(b'engine')
    (tmp_path/'labels').write_text('drone')
    profile=tmp_path/'infer.txt'
    profile.write_text('[property]\nmodel-engine-file=engine\nlabelfile-path=labels\ncustom-lib-path=parser.so\n')
    with pytest.raises(RuntimeError,match='custom-lib-path'): native_profile_assets(tmp_path,profile)
    (tmp_path/'parser.so').write_bytes(b'parser')
    assets=native_profile_assets(tmp_path,profile)
    assert set(assets)=={'model-engine-file','labelfile-path','custom-lib-path'}


def test_rollback_process_race_does_not_abort_restoration(tmp_path):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    manager.names=lambda:['idcs-ui.service']
    props='Id=idcs-ui.service\nMainPID=999999999\nActiveState=activating\n'
    with patch.object(manager,'run',return_value=props):
        state=manager.state('idcs-ui.service')
    assert state['process_cwd'] is None and state['process_cmdline']==''


def test_reset_history_only_for_previously_active_software(tmp_path):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    with patch.object(manager,'run') as run,patch.object(manager,'state',return_value={'NRestarts':'16'}):
        manager.reset_software_failures({'idcs-sim-ui.service':{'ActiveState':'active'},'idcs-hil-streamer.service':{'ActiveState':'inactive'}})
        run.assert_called_once_with(['systemctl','--user','reset-failed','idcs-sim-ui.service'])
    with patch.object(manager,'run') as run:
        with pytest.raises(RuntimeError): manager.reset_software_failures({'idcs-controller.service':{'ActiveState':'active'}})
        run.assert_not_called()


def test_reset_does_not_touch_garbage_collected_zero_state_unit(tmp_path):
    manager=Deployment(tmp_path,'jetson','one',templates=tmp_path)
    with patch.object(manager,'run') as run,patch.object(manager,'state',return_value={'NRestarts':'0'}):
        manager.reset_software_failures({'idcs-recorder.service':{'ActiveState':'active'}})
        run.assert_not_called()


def test_all_canonical_templates_render_and_no_trial_paths():
    root=Path(__file__).resolve().parents[1]/'deploy/systemd'
    for path in root.rglob('*'):
        if path.suffix not in ('.service','.target'): continue
        rendered=render_unit(path.read_text(),Path('/home/idcs/Desktop/project/IDCS-releases/test'))
        assert '@RELEASE@' not in rendered
        assert 'IDCS-trials/' not in rendered and 'IDCS-runtime' not in rendered


def test_old_failure_never_bypasses_successful_activation(tmp_path):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    manager.save('activation.json',{'commit':'success'})
    manager.save('failure.json',{'rolled_back':True})
    with pytest.raises(RuntimeError,match='already activated'): manager.prepare_activation_epoch()
    assert manager.load('activation.json')=={'commit':'success'}


def test_same_unit_hashes_cannot_reuse_tests_from_another_commit(tmp_path):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    manager.save('plan.json',{'commit':'new','units':{}})
    manager.save('validation.json',{'passed':True,'commit':'old','unit_hashes':{}})
    with patch.object(manager,'run') as run,patch.object(manager,'safety') as safety:
        with pytest.raises(RuntimeError,match='validation does not match'): manager.activate()
        run.assert_not_called()
        safety.assert_not_called()


def test_failed_epoch_is_archived_not_lost(tmp_path):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    manager.save('failure.json',{'rolled_back':True,'error':'old'})
    manager.prepare_activation_epoch()
    assert not (manager.evidence/'failure.json').exists()
    assert len(list(manager.evidence.glob('rolled-back-failure-*.json')))==1
    manager.save('failure.json',{'rolled_back':False})
    with pytest.raises(RuntimeError,match='explicit recovery'): manager.prepare_activation_epoch()


@pytest.mark.parametrize('previous_source',['current','older'])
def test_rollback_does_not_leave_false_current_release_claim(tmp_path,previous_source):
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    current=tmp_path/'deployments/managed/pc-current.json'
    current.write_text(json.dumps({'source':str(manager.source) if previous_source=='current' else '/older/release','verified_ns':123}))
    restored={'idcs-ui.service':{'WorkingDirectory':'/retained/trial'}}
    manager.invalidate_current_manifest(restored)
    assert json.loads(current.read_text())['status']=='restored-unverified'
    assert json.loads(current.read_text())['units']==restored
    assert len(list(manager.evidence.glob('pre-rollback-current-*.json')))==1
