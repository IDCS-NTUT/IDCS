"""Offline deployment boundaries; never invokes SSH, systemd or devices."""
from pathlib import Path
import json
from unittest.mock import patch
import pytest
from tools.deployment import (PHYSICAL, Deployment, config_differences, digest,
    require_config_compatible, require_safe_states, render_unit, safe_release,
    native_profile_assets, review_config_files)


@pytest.mark.parametrize('reason',['tracking','coasting','idle_return','target_switch_hold','clock_unavailable'])
def test_swarm_media_health_accepts_target_loss_not_timing_faults(tmp_path,reason):
    import time
    from types import SimpleNamespace
    manager=Deployment(tmp_path,'pc','one',templates=tmp_path)
    states={'idcs-sim-controller.service':{'ActiveState':'inactive'},
            'idcs-sim-swarm-controller.service':{'ActiveState':'active'}}
    record={'ticks':101,'snapshots':21,'last_reason':reason}
    with patch.object(manager,'states',return_value=states),\
         patch.object(Path,'read_text',return_value=json.dumps(record)),\
         patch.object(Path,'stat',return_value=SimpleNamespace(st_mtime=time.time())):
        if reason=='clock_unavailable':
            with pytest.raises(RuntimeError,match='gate failed'):manager.health()
        else:
            assert next(iter(manager.health().values()))==record


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


def test_new_yaml_config_requires_exact_review(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    (old/'configs').mkdir(parents=True)
    (new/'configs').mkdir(parents=True)
    (new/'configs/new.yaml').write_text('controller:\n  mode: live\n')
    with pytest.raises(ValueError, match='unreviewed configuration addition'):
        review_config_files(old, new, {})
    key = f'{old}:{Path("configs/new.yaml")}'
    semantic = lambda value: digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    reviews = {key: {'before': semantic(None), 'after': semantic({'controller': {'mode': 'live'}})}}
    assert review_config_files(old, new, reviews)[key][0][1] == 'changed'
    reviews[key]['after'] = semantic({'controller': {'mode': 'shadow'}})
    with pytest.raises(RuntimeError, match='reviewed config'):
        review_config_files(old, new, reviews)


def test_native_yml_change_requires_byte_review_and_removed_file_fails(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    (old/'configs').mkdir(parents=True)
    (new/'configs').mkdir(parents=True)
    old_file = old/'configs/tracker.yml'
    new_file = new/'configs/tracker.yml'
    old_file.write_bytes(b'%YAML:1.0\nvalue: 1\n')
    new_file.write_bytes(b'%YAML:1.0\nvalue: 2\n')
    with pytest.raises(ValueError, match='unreviewed configuration drift'):
        review_config_files(old, new, {})
    key = f'{old}:{Path("configs/tracker.yml")}'
    reviews = {key: {'before': digest(old_file.read_bytes()), 'after': digest(new_file.read_bytes())}}
    assert review_config_files(old, new, reviews)[key][0][1] == 'changed'
    new_file.unlink()
    with pytest.raises(RuntimeError, match='missing configuration'):
        review_config_files(old, new, reviews)


def test_removed_config_needs_exact_removal_review(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    (old/'configs').mkdir(parents=True)
    (new/'configs').mkdir(parents=True)
    (old/'configs/trial.yaml').write_text('sim:\n  scene: x\n')
    key = f'{old}:{Path("configs/trial.yaml")}'
    semantic = lambda value: digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    with pytest.raises(RuntimeError, match='missing configuration'):
        review_config_files(old, new, {})
    with pytest.raises(RuntimeError, match='missing configuration'):
        review_config_files(old, new, {key: {'before': semantic({'sim': {'scene': 'x'}}), 'after': None}})
    with pytest.raises(RuntimeError, match='missing configuration'):
        review_config_files(old, new, {key: {'removed': True, 'before': semantic({'sim': {'scene': 'y'}})}})
    reviews = {key: {'removed': True, 'before': semantic({'sim': {'scene': 'x'}})}}
    assert review_config_files(old, new, reviews)[key][0][1] == 'removed'


def test_shared_gate_bootstraps_only_without_any_live_state(tmp_path, monkeypatch):
    from tools.deployment import shared_config_gate
    live = tmp_path/'idcs/live.yaml'
    monkeypatch.setenv('IDCS_LIVE_CONFIG', str(live))
    assert shared_config_gate({})['bootstrap']
    live.parent.mkdir()
    live.write_text('{}\n')
    with pytest.raises(RuntimeError, match='status unavailable'):
        shared_config_gate({})
    live.unlink()
    (live.parent/'config-shared.json').write_text('{"ready": false}')
    with pytest.raises(RuntimeError, match='unreconciled'):
        shared_config_gate({})
    with pytest.raises(RuntimeError, match='status unavailable'):
        shared_config_gate(None, marker=tmp_path/'absent.json')


def test_pi_stage_skips_shared_gate_unless_evidence_is_given(tmp_path, monkeypatch):
    monkeypatch.setenv('IDCS_LIVE_CONFIG', str(tmp_path/'idcs/live.yaml'))
    (tmp_path/'idcs').mkdir()
    (tmp_path/'idcs/config-shared.json').write_text('{"ready": false}')
    manager = Deployment(tmp_path, 'pi', 'one', templates=tmp_path)
    with pytest.raises(ValueError, match='full explicit commit'):
        manager.stage('bundle', 'short')  # gate passed, refused later on the commit id
    with pytest.raises(RuntimeError, match='unreconciled'):
        manager.stage('bundle', 'short', shared_status=tmp_path/'idcs/config-shared.json')


def test_new_native_yml_profile_also_requires_review(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    (old/'configs').mkdir(parents=True)
    (new/'configs').mkdir(parents=True)
    profile = new/'configs/nvdcf.yml'
    profile.write_bytes(b'%YAML:1.0\nvalue: 1\n')
    with pytest.raises(ValueError, match='unreviewed configuration drift'):
        review_config_files(old, new, {})
    key = f'{old}:{Path("configs/nvdcf.yml")}'
    reviews = {key: {'before': digest(b''), 'after': digest(profile.read_bytes())}}
    assert review_config_files(old, new, reviews)[key][0][1] == 'added'


def test_plan_rejects_new_overlay_before_unit_or_systemd_work(tmp_path):
    manager = Deployment(tmp_path, 'pc', 'one', templates=tmp_path/'templates')
    prior = tmp_path/'IDCS-releases/previous'
    (prior/'configs').mkdir(parents=True)
    (manager.source/'configs').mkdir(parents=True)
    (manager.source/'configs/new.yaml').write_text('controller:\n  yaw_kp: 20\n')
    inventory = {'units': {'idcs-test.service': {
        'state': {'WorkingDirectory': str(prior)}, 'dropins': {},
    }}}
    with patch.object(manager, 'inventory', return_value=inventory), \
         patch.object(manager, 'names', return_value=['idcs-test.service']), \
         patch.object(manager, 'run') as run:
        with pytest.raises(ValueError, match='unreviewed configuration addition'):
            manager.plan()
        run.assert_not_called()


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


def test_jetson_camera_target_restored_before_services_but_hil_refused(tmp_path):
    manager=Deployment(tmp_path,'jetson','one',templates=tmp_path)
    previous={'idcs-camera.target':{'ActiveState':'active'},
              'idcs-deepstream-camera.service':{'ActiveState':'active'}}
    with patch.object(manager,'run') as run, patch.object(manager,'state',
            side_effect=lambda name: {'ActiveState':'inactive' if name.endswith('.target') else 'active'}):
        manager.restart_previous(previous)
        run.assert_called_once_with(['sudo','-n','systemctl','start','idcs-camera.target'])
        with pytest.raises(RuntimeError,match='physical activation'):
            manager.restart_previous({'idcs-hil.target':{'ActiveState':'active'}})
        run.assert_called_once()


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
