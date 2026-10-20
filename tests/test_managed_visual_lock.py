"""Managed console contracts: deterministic fakes, no GPU, systemd or lab hosts."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from common.config import ConfigError, load_config_bundle, resolve_config_paths
from common.operator_commands import OperatorCommand
from common.return_frame_identity import RoiSourceLease
from jetson.operator_agent import AgentConfig
from jetson.visual_lock.live_trial import LiveDeadline, resolve_snapshot_bind
from jetson.visual_lock.managed import BoundedLog, evidence_directory, recycle_lost_roi, validate_managed
from rpi.display.menu import Settings, NavEvent
from rpi.display.roi_editor import RoiEditor, ROI_SETTINGS, BOX_SIZES
from rpi.display.roi_panel import RoiPanelWorkflow
from tests.test_operator_agent import _agent, config, _snapshot
from tests.test_visual_lock_acquisition import coordinator, replay, BOX
from jetson.visual_lock.acquisition import acquire_cached
from tests.test_visual_lock import frame
from tools.deployment import Deployment, MANAGED_VISUAL_UNITS, visual_lock_assets
from tools.sim_panel import RoiSimAuthority, armed_state, disabled_state

ROOT = Path(__file__).resolve().parents[1]
MODES = ('camera_visual_lock','sim_visual_lock')


@pytest.mark.parametrize('mode',MODES)
def test_modes_transition_start_stop_ready_and_back(config, mode):
    agent, systemd, _ = _agent(config)
    assert agent.handle(OperatorCommand('set_mode',mode=mode))[0]
    agent.wait_idle()
    assert agent.current_mode() == mode and agent.state()['visual_lock']
    expected = config.mode(mode)
    assert [row for row in systemd.trace if row[0:2] == ('Jetson','start')] == [
        ('Jetson','start',unit) for unit in expected.start]
    assert agent.pc.active_units >= set(expected.pc_ready_units)
    assert not {'idcs-deepstream-camera.service','idcs-deepstream-video.service'} & systemd.active
    trace = list(systemd.trace)
    assert all(row[1] == 'stop' for row in trace[:len(expected.pc_stop)+len(expected.stop)])
    assert agent.handle(OperatorCommand('set_mode',mode='sim'))[0]
    agent.wait_idle()
    assert agent.current_mode() == 'sim'
    assert not set(MANAGED_VISUAL_UNITS) & systemd.active


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('failure',('start','ready','pc_start'))
def test_failed_visual_transition_rolls_back_exact_previous_mode(config,mode,failure):
    config = replace(config,readiness_timeout_s=.01)
    agent, systemd, _ = _agent(config)
    if failure == 'start':
        agent.run = lambda args,**kw: (1,'injected') if args == ['start',config.mode(mode).start[0]] else systemd(args,**kw)
    elif failure == 'ready':
        agent.ready_probe = lambda incoming,*_: incoming.name != mode
    else:
        original = agent.pc.run
        # Fail once; rollback must still be able to restart its previous PC source.
        failed = []
        def fail(args,**kw):
            if args == ['start',config.mode(mode).pc_start[0]] and not failed:
                failed.append(True)
                return 1,'injected'
            return original(args,**kw)
        agent.pc.run = fail
    assert agent.handle(OperatorCommand('set_mode',mode=mode))[0]
    agent.wait_idle()
    assert agent.current_mode() == 'sim'
    assert 'restored Simulator' in agent.last_message
    assert not set(MANAGED_VISUAL_UNITS) & systemd.active


@pytest.mark.parametrize('mode',MODES)
def test_managed_readiness_uses_mode_health_and_matching_session(config,tmp_path,mode):
    agent,_,clock = _agent(config)
    incoming = replace(config.mode(mode),health_file=tmp_path/'health.json')
    agent.ready_probe = None
    agent._transition_s = clock.t-1
    snapshot = _snapshot([]).model_dump(mode='json')
    snapshot['frame'].update(tracker_session='roi-session',source_clock_domain=incoming.source_clock,
        observation_clock_domain='jetson_monotonic',observed_time_ns=int(clock.t*1e9))
    for i in range(3):
        snapshot['frame'].update(frame_id=i,source_time_ns=i+1)
        agent.on_snapshot(snapshot)
    health = dict(managed_mode=mode,tracker_session='roi-session',last_frame_monotonic_s=clock.t,frames=3,return_frames=1)
    def check(**changes):
        incoming.health_file.write_text(json.dumps(dict(health,**changes)))
        return agent._video_ready(incoming,clock.t-1)
    assert check()
    assert not check(tracker_session='previous')
    assert not check(managed_mode='sim' if mode == 'sim_visual_lock' else 'camera')
    assert not check(return_frames=0)
    assert not check(last_frame_monotonic_s=clock.t-2)
    assert not check(frames=2)
    agent._fresh_frames = 2
    assert not check()


@pytest.mark.parametrize('mode',MODES)
def test_detector_authority_and_expiry_cannot_interfere(config,mode):
    agent,_,clock = _agent(config)
    assert agent.handle(OperatorCommand('set_mode',mode=mode))[0]
    agent.wait_idle()
    agent.on_snapshot(_snapshot([(7,'visual')]).model_dump(mode='json'))
    agent.pc.targets = NS(forward=lambda _: (_ for _ in ()).throw(AssertionError('truth catalog accessed')))
    assert agent.active_catalog() is None
    agent.selection_mode = 'operator'
    before = agent.sequence
    clock.t += config.lock_lost_s+1
    agent.expire_lock()
    assert agent.lock_track_id is None and agent.sequence == before
    for command in (OperatorCommand('lock',track_id=7),OperatorCommand('release'),
                    OperatorCommand('selection_mode',mode='planner'),OperatorCommand('target_classes',classes=('drone',)),
                    OperatorCommand('clear_swarm_score',source_session='old')):
        assert not agent.handle(command)[0]


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('field,value', [('start',['idcs-deepstream-video.service']),('health_file','relative.json'),
    ('source_clock','wrong'),('ready_units',[]),('stop',[]),('pc_start',['idcs-sim.target'])])
def test_mode_config_refuses_invalid_visual_owner(mode,field,value):
    raw=load_config_bundle(resolve_config_paths('configs/base')).mutable_copy()
    raw['operator']['modes'][mode][field]=value
    with pytest.raises(ConfigError): AgentConfig.from_config(raw)


def test_managed_port_requires_explicit_switch_for_every_trial():
    for panel in (True,False):
        assert resolve_snapshot_bind(None,panel_roi=panel).endswith(':5597')
        for host in ('0.0.0.0','127.0.0.1','*','[::]'):
            with pytest.raises(ValueError): resolve_snapshot_bind(f'tcp://{host}:5564',panel_roi=panel)
        for mode in MODES:
            assert resolve_snapshot_bind(None,panel_roi=panel,managed_mode=mode).endswith(':5564')
    with pytest.raises(ValueError): resolve_snapshot_bind(None,panel_roi=True,managed_mode='invalid')
    with pytest.raises(ValueError): resolve_snapshot_bind('tcp://*:5597',panel_roi=True,managed_mode=MODES[0])
    validate_managed(MODES[0],'camera',roi_bind='roi',identity_bind='identity',seed=None,handoff=False)
    with pytest.raises(ValueError): validate_managed(MODES[0],'sim',roi_bind='roi',identity_bind='identity',seed=None,handoff=False)
    with pytest.raises(ValueError): validate_managed(MODES[0],'camera',roi_bind='roi',identity_bind='identity',seed=(1,2,3,4),handoff=False)
    deadline=LiveDeadline(1,0,5)
    assert deadline.reason(6) == 'source startup timeout'
    deadline.observe(2)
    assert deadline.reason(1e6) is None


def test_real_session_loss_recycles_resources_then_requires_explicit_new_box():
    session, backends = coordinator()
    lock=acquire_cached(session,replay(0,1,2),BOX)
    assert lock.result.snapshot.selection
    workflow=RoiPanelWorkflow(active=True,state='tracking',request_id='box')
    lost=session.update(frame(30),object())  # actual source-gap loss
    assert lost.decision.state == 'lost' and lost.snapshot.selection is None
    cache, prepared, notices = Mock(),[],[]
    assert recycle_lost_roi('box',lost,close=session.close,clear_cache=cache.clear,
        prepare=lambda:prepared.append('fresh-backend'),notify=lambda request,accepted,reason:notices.append((request,accepted,reason)))
    assert backends[-1].closed and prepared == ['fresh-backend']
    cache.clear.assert_called_once()
    assert workflow.result(notices[0][0],notices[0][1])
    assert workflow.request_id is None
    assert not session.update(frame(31),object()).snapshot.selection
    assert len(backends) == 1  # no automatic new tracker seed
    fresh=acquire_cached(session,replay(40,41,42),BOX)
    assert fresh.accepted and fresh.result.snapshot.selection and len(backends)==2
    refused=acquire_cached(session,[(frame(50),'blank')],BOX)
    assert not refused.accepted and refused.result.snapshot.selection is None
    assert workflow.state == 'camera'  # fresh physical OFF still required


def test_box_size_persistence_default_clamping_and_editor_open(tmp_path):
    path=tmp_path/'operator_display.json'
    settings=Settings(ROI_SETTINGS,path)
    assert settings['roi_box_size']=='Medium'
    settings.step('roi_box_size')
    assert Settings(ROI_SETTINGS,path)['roi_box_size']=='Large'
    path.write_text('{"roi_box_size":"unknown"}')
    assert Settings(ROI_SETTINGS,path)['roi_box_size']=='Medium'
    for size,(w,h) in BOX_SIZES.items():
        editor=RoiEditor()
        editor.open(1280,720,direct_select=True,box_size=size)
        assert (editor.w,editor.h)==(w,h)
        editor.open(40,30,box_size=size)
        assert editor.w<=40 and editor.h<=30 and editor.x>=0 and editor.y>=0


def test_managed_lease_requires_exact_mode_source_and_session():
    payload=dict(type='visual_roi_status_v1',source_mode='sim',managed_mode='sim_visual_lock',
        control_disabled=False,source_ready=True,tracker_session='run',sequence=1)
    lease=RoiSourceLease()
    assert not lease.observe(payload,1.)  # isolated display cannot consume managed authority
    assert not lease.observe(payload,1.,managed_mode='camera_visual_lock')
    plain=dict(payload,managed_mode=None,control_disabled=True)
    assert not lease.observe(plain,1.,managed_mode='sim_visual_lock')
    assert lease.observe(plain,1.)
    payload['sequence']=2
    assert lease.observe(payload,1.,managed_mode='sim_visual_lock')
    assert lease.live_session(1.2,'run') and not lease.live_session(1.6,'run')


def test_evidence_is_bounded_without_pruning(tmp_path):
    log=BoundedLog(tmp_path/'frames.jsonl',max_bytes=10)
    log.write('1234\n'); log.write('5678\n'); log.write('discarded\n'); log.close()
    assert (tmp_path/'frames.jsonl').read_text()=='1234\n5678\n'
    assert log.report()['dropped']==1
    root=tmp_path/'sessions'
    for _ in range(8): evidence_directory(root).mkdir()
    with pytest.raises(ValueError,match='retention full'): evidence_directory(root)
    assert len(list(root.iterdir()))==8


def test_native_presence_check_does_not_load_gpu(tmp_path):
    with pytest.raises(RuntimeError,match='native asset'): visual_lock_assets(tmp_path)
    build=tmp_path/'build/visual-lock';build.mkdir(parents=True)
    for name in ('libidcs_visual_lock.so','libidcs_visual_lock_live_surface.so'): (build/name).write_bytes(b'fixture')
    assert len(visual_lock_assets(tmp_path))==2


def test_hardware_units_never_load_sim_overlay():
    for name in ('idcs-controller','idcs-bridge','idcs-serial','idcs-controller-camera'):
        text=(ROOT/f'deploy/systemd/jetson/{name}.service').read_text()
        assert 'configs/sim/' not in text, name


def test_new_units_are_planned_without_boot_or_hardware_activation(tmp_path,monkeypatch):
    manager=Deployment(tmp_path,'jetson','one',templates=ROOT/'deploy/systemd')
    # Model an existing install with every old unit but neither new fragment.
    manager.unitdir=tmp_path/'units'
    inventory={'units':{name:dict(state={'ActiveState':'inactive'},main_path=str(manager.unitdir/name),
        main_text='old',dropins={}) for name in manager.names() if name not in MANAGED_VISUAL_UNITS}}
    monkeypatch.setattr(manager,'inventory',lambda:inventory)
    monkeypatch.setattr(manager,'state',lambda name:dict(ActiveState='inactive',FragmentPath=''))
    monkeypatch.setattr(manager,'run',lambda *a,**kw:'a'*40)
    plan=manager.plan()
    assert plan['units']==len(manager.names())
    saved=manager.load('plan.json')
    for name in MANAGED_VISUAL_UNITS:
        unit=saved['units'][name]
        assert unit['old']['main_text'] is None and '--managed-mode ' in unit['desired']
        assert '[Install]' not in unit['desired'] and 'configs/sim/' not in unit['desired']
        assert 'idcs-bridge' not in unit['desired'] and 'idcs-serial' not in unit['desired']
        assert '--source '+('camera' if 'camera' in name else 'sim') in unit['desired']
    inventory['units']['idcs-unknown.service']=dict(state={},main_path='x',main_text='',dropins={})
    with pytest.raises(RuntimeError,match='inventory differs'): manager.plan()


def test_no_detector_class_or_range_aims_at_camera_centre_without_parallax():
    from common.aiming import solve_snapshot_aiming
    from common.control import ControlConfig,LaserMountConfig
    raw=load_config_bundle(resolve_config_paths('configs/base')).mutable_copy()
    raw['control']['aim_mode']='laser_point'
    control=ControlConfig.from_raw_config(raw,(1280,720))
    mount=LaserMountConfig.from_raw_config(raw)
    # V2 requires a class label. Production uses 'visual', with no detector
    # class/known-size assessment. Exercise the actual schema-valid snapshot.
    from common.perception import PerceptionSnapshotV2
    session,_=coordinator()
    output=acquire_cached(session,replay(0,1,2),BOX).result
    snapshot=PerceptionSnapshotV2.model_validate_json(output.snapshot.model_dump_json())
    assert snapshot.tracks[0].class_id == 'visual' and not snapshot.assessments
    aim=solve_snapshot_aiming(snapshot,control,mount)
    assert aim and aim.class_id == 'visual' and aim.distance_m is None and not aim.parallax_active
    scaled=control.for_frame_size((snapshot.frame.width,snapshot.frame.height))
    assert aim.aim_px==(scaled.cx_px,scaled.cy_px)


def test_sim_only_lease_needs_fresh_panel_receiver_and_request():
    roi=RoiSimAuthority()
    panel=armed_state().model_copy(update=dict(source='rpi.runtime_control',active=True,control_cmd_enabled=False))
    lease=dict(type='roi_sim_lease_v1',enabled=True,managed_mode='sim_visual_lock',request_id='box',tracker_session='run')
    status=dict(type='visual_roi_status_v1',source_mode='sim',managed_mode='sim_visual_lock',source_ready=True,
        control_disabled=False,active_request_id='box',tracker_session='run',sequence=1)
    roi.observe(json.dumps(lease),1.)
    assert not roi.apply(panel,1.01).control_cmd_enabled
    roi.observe(json.dumps(status),1.,status=True)
    applied=roi.apply(panel,1.01)
    assert applied.control_cmd_enabled and not applied.active and applied.joystick_rate_cmd==(0.,0.)
    assert not roi.apply(panel.model_copy(update={'emergency':True}),1.01).control_cmd_enabled
    assert not roi.apply(disabled_state('stale panel'),1.01).control_cmd_enabled
    assert not roi.apply(panel,1.26).control_cmd_enabled
    status['active_request_id']='other';status['sequence']=2
    roi.observe(json.dumps(status),1.01,status=True)
    assert not roi.apply(panel,1.02).control_cmd_enabled
    lease['enabled']=False; roi.observe(json.dumps(lease),1.02)
    assert not roi.apply(panel,1.03).control_cmd_enabled


@pytest.fixture
def display_module(monkeypatch):
    # Import only the real menu/workflow code; rendering/Gst are not under test.
    gi=ModuleType('gi');gi.require_version=lambda *a:None
    repository=ModuleType('gi.repository');repository.GLib=NS();repository.Gst=NS()
    monkeypatch.setitem(sys.modules,'gi',gi);monkeypatch.setitem(sys.modules,'gi.repository',repository)
    for name,attrs in {'rpi.display.render':('StatusBar','TargetBox','render'),
                       'rpi.display.video':('ReturnVideo','missing_elements'),
                       'rpi.display.hud':('PanelHudState','PiHudRenderer')}.items():
        module=ModuleType(name)
        for attr in attrs: setattr(module,attr,Mock())
        monkeypatch.setitem(sys.modules,name,module)
    spec=importlib.util.spec_from_file_location('managed_display_contract',ROOT/'rpi/operator_display.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode',MODES)
def test_normal_menu_visual_items_replace_detector_selection(display_module,mode):
    from rpi.display.status import SystemStatus
    status=SystemStatus();status.agent_state=dict(mode=mode,sim_visual_lock=True,sim_scenes=[dict(name='swarm',label='Swarm')],
        modes=[dict(name='camera',label='Camera'),dict(name='sim',label='Simulator'),dict(name='camera_visual_lock',label='Camera / Visual lock')])
    select,release=Mock(),Mock()
    menu=display_module.build_menu(status,Settings(ROI_SETTINGS,None),NS(),roi_preview=select,roi_release=release)
    targets=menu.root.current_items()[0]
    items=targets.current_items()
    assert [item.label for item in items]==['Select box','Release','Box size']
    items[0].run();items[1].run();select.assert_called_once();release.assert_called_once()
    operations=menu.root.current_items()[1]
    source=operations.current_items()[0]
    simulator=source.current_items()[1]
    assert [item.label for item in simulator.current_items()]==['Swarm','Visual lock']
    assert 'Clear Swarm score' not in [getattr(item,'label',None) for item in operations.current_items()]
    status.agent_state['mode']='camera'
    assert [getattr(item,'label',getattr(item,'title',None)) for item in targets.current_items()]==['Planner selection','Eligible classes']


def test_managed_display_opens_persisted_size_and_handles_terminal_status(display_module):
    display=display_module.OperatorDisplay.__new__(display_module.OperatorDisplay)
    display.visual_roi_trial=False
    display.status=NS(agent_state={'mode':'sim_visual_lock'},show_notice=Mock())
    display.settings=Settings(ROI_SETTINGS,None);display.settings.step('roi_box_size')
    display.menu=NS(_close=Mock());display.roi_editor=RoiEditor();display.roi_workflow=RoiPanelWorkflow()
    display.roi_lamp_pub=None
    display.video=NS(frame_identity=NS(latest=lambda:object()),frame_size=(1280,720))
    display._roi_preview_allowed=lambda _:True
    display._start_roi_preview()
    assert (display.roi_editor.w,display.roi_editor.h)==(160,120)
    assert display.roi_editor.direct_select and display.roi_workflow.active
    display.roi_workflow.state='tracking';display.roi_workflow.request_id='box'
    display._roi_request_session='run';display._pending_roi_request=None
    display._roi_terminal('box','lost texture',1.)
    assert display.roi_workflow.request_id is None and display._roi_request_session is None
    assert display.roi_editor.active


@pytest.mark.parametrize('installed',(False,True))
def test_new_unit_rollback_handles_failure_before_or_after_install(tmp_path,monkeypatch,installed):
    manager=Deployment(tmp_path,'jetson','one',templates=ROOT/'deploy/systemd')
    name='idcs-visual-lock-camera.service'
    target=tmp_path/name
    if installed: target.write_text('candidate')
    prior={name:{'ActiveState':'inactive'}}
    manager.save('rollback.json',dict(previous=prior,units={name:dict(main_path=str(target),main_text=None,dropins={})}))
    manager.save('plan.json',dict(units={name:dict(desired='candidate')}))
    monkeypatch.setattr(manager,'safety',Mock())
    monkeypatch.setattr(manager,'states',lambda:prior)
    monkeypatch.setattr(manager,'restart_previous',Mock())
    monkeypatch.setattr(manager,'invalidate_current_manifest',Mock())
    run=Mock();monkeypatch.setattr(manager,'run',run)
    assert manager.rollback()['restored']
    removals=[call.args[0] for call in run.call_args_list if 'rm' in call.args[0]]
    assert removals==([['sudo','-n','rm','--',str(target)]] if installed else [])


def test_check_cli_is_static_and_plain_trial_port_refusal_precedes_gpu(tmp_path,monkeypatch):
    import jetson.visual_lock.live_trial as live
    library=tmp_path/'lib.so';library.write_bytes(b'presence-only')
    run=Mock(side_effect=AssertionError('GPU run called'))
    monkeypatch.setattr(live,'run',run)
    arguments=['--library',str(library),'--surface-library',str(library),'--output',str(tmp_path/'evidence'),
        '--roi-bind','tcp://*:5596','--return-identity-bind','tcp://*:5595','--return-host','127.0.0.1',
        '--provenance','unit-test','--check']
    for mode in MODES:
        assert live.main(arguments+['--managed-mode',mode,'--source',mode.removesuffix('_visual_lock'),
            '--duration-s','0','--health-file',str(tmp_path/'health.json')])==0
    with pytest.raises(SystemExit) as refusal:
        live.main(arguments+['--snapshot-bind','tcp://*:05564'])
    assert refusal.value.code==2
    run.assert_not_called()


def test_managed_status_keeps_recording_and_hides_detector_planner():
    from rpi.display.status import SystemStatus
    status=SystemStatus()
    status.on_agent(dict(mode='sim_visual_lock',selection_mode='planner',recording=True),1.)
    assert 'VISUAL LOCK' in status.agent_text(1.1) and 'REC' in status.agent_text(1.1)
    assert 'PLANNER' not in status.agent_text(1.1)


def test_receiver_status_and_ack_are_read_without_invalidating_source(display_module,monkeypatch):
    from rpi.display.status import SystemStatus
    display=display_module.OperatorDisplay.__new__(display_module.OperatorDisplay)
    display.visual_roi_trial=False
    display.status=SystemStatus();display.status.on_agent(dict(mode='sim_visual_lock'),1.)
    display.roi_workflow=RoiPanelWorkflow(active=True,state='pending',request_id='box')
    display.roi_source=RoiSourceLease();display._pending_roi_request='box';display._roi_request_session='run'
    status=dict(type='visual_roi_status_v1',managed_mode='sim_visual_lock',source_mode='sim',control_disabled=False,
        source_ready=True,tracker_session='run',sequence=1)
    ack=dict(type='visual_roi_result_v1',accepted=True,request_id='box')
    messages=[(status,1.),(ack,1.)]
    display.identity_receiver=NS(drain=lambda:iter(messages))
    for name in ('perception','roi_perception','diagnostics','pose','hud_sim','panel'):setattr(display,name,None)
    monkeypatch.setattr(display_module.time,'monotonic',lambda:1.01)
    display._read_messages(1.01)
    assert display.roi_source.live_session(1.01,'run') and display.roi_workflow.state=='tracking'
    messages[:]=[(dict(status,sequence=2,terminal_request_id='box',reason='lost'),1.02)]
    monkeypatch.setattr(display_module.time,'monotonic',lambda:1.03)
    display._read_messages(1.03)
    assert display.roi_workflow.request_id is None and display._roi_request_session is None


def test_managed_modes_are_known_restart_config_fields():
    from common.config_schema import validate_fields,tier
    raw=load_config_bundle(resolve_config_paths('configs/base')).mutable_copy()
    validate_fields(raw)
    for mode in MODES:
        assert tier(f'operator.modes.{mode}.health_file')=='restart'


@pytest.mark.parametrize('source',('camera','sim'))
def test_deployment_health_requires_fresh_managed_video(tmp_path,monkeypatch,source):
    import time
    manager=Deployment(tmp_path,'jetson','one')
    states={name:{'ActiveState':'inactive'} for name in ('idcs-deepstream-camera.service','idcs-deepstream-video.service')}
    states[f'idcs-visual-lock-{source}.service']={'ActiveState':'active'}
    monkeypatch.setattr(manager,'states',lambda:states)
    record=dict(managed_mode=source+'_visual_lock',tracker_session='run',last_frame_monotonic_s=time.monotonic(),frames=10,return_frames=3)
    monkeypatch.setattr(Path,'read_text',lambda *a,**kw:json.dumps(record))
    monkeypatch.setattr(Path,'stat',lambda *a,**kw:NS(st_mtime=time.time()))
    assert manager.health()
    record['return_frames']=0
    with pytest.raises(RuntimeError,match='gate failed'): manager.health()
