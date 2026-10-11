"""Explicit, scoped release deployment. No motor/HIL activation or deletion.

Run this file directly on a host; it needs only the standard library until
validation. Installed state lives outside the source checkout. Unit templates
use @RELEASE@; advancing a Git checkout is not deployment or activation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PHYSICAL = frozenset(('idcs-controller.service', 'idcs-bridge.service',
    'idcs-serial.service', 'idcs-hil.target', 'idcs-hil-streamer.service'))
KNOWN_OVERRIDE = 'nested-boxes-20261007.conf'
CONFIG_ADDITIONS = frozenset(('operator.hud', 'rpi.operator_display.hud_sim_bind'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_release(root, release):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', release):
        raise ValueError('invalid release identifier')
    path = root / 'IDCS-releases' / release
    if path.is_symlink() or path.resolve().parent != (root / 'IDCS-releases').resolve():
        raise ValueError('release must be a direct non-symlink child')
    return path


def render_unit(text, source):
    text = text.replace('\r\n', '\n')
    if 'IDCS-trials/' in text or 'IDCS-runtime' in text or 'IDCS-releases/' in text:
        raise ValueError('template contains a fixed/obsolete source path')
    rendered = text.replace('@RELEASE@', str(source))
    if '@RELEASE@' in rendered:
        raise ValueError('unresolved source token')
    if '[Service]' in text and f'WorkingDirectory={source}\n' not in rendered:
        raise ValueError('service must declare the release working directory')
    return rendered


def config_differences(old, new, prefix=''):
    """Only explicitly reviewed read-only HUD additions may change by default."""
    changes = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            name = f'{prefix}.{key}' if prefix else str(key)
            if key not in old:
                changes.append((name, 'added', new[key]))
            elif key not in new:
                changes.append((name, 'removed', old[key]))
            else:
                changes.extend(config_differences(old[key], new[key], name))
    elif old != new:
        changes.append((prefix, 'changed', [old, new]))
    return changes


def require_config_compatible(old, new):
    differences = config_differences(old, new)
    unexpected = [c for c in differences if not (c[1] == 'added' and c[0] in CONFIG_ADDITIONS)]
    if unexpected:
        raise ValueError('unreviewed configuration drift: ' + repr(unexpected))
    return differences


def review_config_files(previous, candidate, reviews):
    """Compare the full config-file union; new files require explicit review."""
    import yaml
    previous, candidate = Path(previous), Path(candidate)
    files = lambda root: {p.relative_to(root) for p in (root/'configs').rglob('*')
                          if p.is_file() and p.suffix.lower() in ('.yaml', '.yml')}
    old_files, new_files = files(previous), files(candidate)
    changes = {}
    for rel in sorted(old_files | new_files):
        old, new = previous/rel, candidate/rel
        if rel not in new_files:
            raise RuntimeError(f'missing configuration: {rel}')
        key = f'{previous}:{rel}'
        reviewed = reviews.get(key)
        if rel.suffix.lower() == '.yml':
            # Native GStreamer/NvDCF profiles may not be valid PyYAML. Their
            # exact bytes, not an attempted semantic parse, are the contract.
            before = old.read_bytes() if rel in old_files else b''
            after = new.read_bytes()
            if before == after and rel in old_files:
                continue
            if not reviewed or reviewed['before'] != digest(before) or reviewed['after'] != digest(after):
                raise ValueError('unreviewed configuration drift: '+key)
            changes[key] = [(str(rel), 'added' if rel not in old_files else 'changed', digest(after))]
            continue
        old_cfg = yaml.safe_load(old.read_text()) if rel in old_files else None
        new_cfg = yaml.safe_load(new.read_text())
        semantic = lambda value: digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
        if rel not in old_files and not reviewed:
            raise ValueError('unreviewed configuration addition: '+key)
        if reviewed:
            if reviewed['before'] != semantic(old_cfg) or reviewed['after'] != semantic(new_cfg):
                raise RuntimeError('reviewed config bytes/values changed: '+key)
            changes[key] = config_differences(old_cfg, new_cfg)
        else:
            changes[key] = require_config_compatible(old_cfg, new_cfg)
    return {key: value for key, value in changes.items() if value}


def require_safe_states(states):
    unsafe = {n:s for n,s in states.items() if n in PHYSICAL and s.get('ActiveState') != 'inactive'}
    if unsafe:
        raise RuntimeError('physical/HIL stack is not inactive: ' + repr(unsafe))


def native_profile_assets(source, profile):
    """Static prerequisite checks: no GPU/engine deserialization or pipeline."""
    import configparser
    config=configparser.ConfigParser(interpolation=None)
    config.read(profile)
    records={}
    for key in ('model-engine-file','labelfile-path','custom-lib-path'):
        value=config['property'][key]
        path=Path(value) if Path(value).is_absolute() else source/value
        if not path.is_file(): raise RuntimeError(f'missing nvinfer {key}: {path}')
        records[key]=dict(path=str(path),resolved=str(path.resolve()),sha256=digest(path.read_bytes()))
    return records


class Deployment:
    def __init__(self, root, role, release, templates=None, config_policy=None):
        self.root, self.role, self.release = Path(root), role, release
        self.source = safe_release(self.root, release)
        self.evidence = self.root / 'deployments' / 'managed' / release / role
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.templates = Path(templates) if templates else self.source / 'deploy/systemd'
        self.config_policy=Path(config_policy) if config_policy else None
        self.unitdir = Path('/etc/systemd/system') if role == 'jetson' else Path.home()/'.config/systemd/user'
        self.ctl = ['sudo','-n','systemctl'] if role=='jetson' else ['systemctl','--user']
        self.python = self.root / ('venv/bin/python3' if role == 'pc' else 'bin/python3')

    def run(self, args, **kwargs):
        result = subprocess.run([str(x) for x in args], text=True, capture_output=True,
            timeout=kwargs.pop('timeout', 120), **kwargs)
        if result.returncode:
            raise RuntimeError(f'{args}: {result.stdout}\n{result.stderr}')
        return result.stdout + result.stderr

    def save(self, name, value):
        path = self.evidence / name
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(path)

    def load(self, name):
        return json.loads((self.evidence/name).read_text())

    def state(self, name):
        result = self.run(self.ctl+['show',name,'-p','Id','-p','ActiveState','-p','MainPID',
            '-p','NRestarts','-p','WorkingDirectory','-p','FragmentPath','-p','DropInPaths'])
        state = dict(line.split('=',1) for line in result.splitlines() if '=' in line)
        pid = state.get('MainPID', '0')
        if pid not in ('0',''):
            proc = Path('/proc')/pid
            try:
                state['process_cwd'] = str((proc/'cwd').resolve())
                state['process_cmdline'] = (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            except (FileNotFoundError,ProcessLookupError):
                state['process_cwd']=None
                state['process_cmdline']=''
        return state

    def names(self):
        folder = self.templates / ('rpi' if self.role=='pi' else self.role)
        return sorted(p.name for p in folder.iterdir() if p.suffix in ('.service','.target'))

    def states(self):
        return {n:self.state(n) for n in self.names()}

    def safety(self):
        names = PHYSICAL if self.role == 'jetson' else ('idcs-hil-streamer.service',) if self.role=='pc' else ()
        states = {n:self.state(n) for n in names}
        require_safe_states(states)
        return states

    def inventory(self):
        units = {}
        # Include ALL installed IDCS fragments, not just currently running ones.
        for path in sorted(self.unitdir.glob('idcs-*')):
            if not path.is_file():
                continue
            state = self.state(path.name)
            drops = {p:Path(p).read_text() for p in state.get('DropInPaths','').split()}
            units[path.name] = dict(state=state, main_path=str(path), main_text=path.read_text(), dropins=drops)
        inventory = dict(role=self.role, units=units, safety=self.safety(), captured_ns=time.time_ns())
        self.save('inventory.json', inventory)
        return inventory

    def stage(self, bundle, commit):
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('full explicit commit required')
        self.safety()
        runtime = self.root/'IDCS-runtime'
        if not self.source.exists():
            self.run(['git','-C',runtime,'bundle','verify',bundle])
            self.run(['git','-C',runtime,'fetch',bundle,f'refs/tags/{self.release}:refs/tags/{self.release}'])
            self.run(['git','-C',runtime,'worktree','add','--detach',self.source,commit])
        actual = self.run(['git','-C',self.source,'rev-parse','HEAD']).strip()
        if actual != commit or self.run(['git','-C',self.source,'status','--porcelain','--untracked-files=no']).strip():
            raise RuntimeError('release is not the expected clean commit')
        self.save('stage.json',dict(source=str(self.source),commit=commit,
            bundle_sha256=digest(Path(bundle).read_bytes()), service_activation=False))
        return self.plan()

    def plan(self):
        inventory = self.inventory()
        if set(inventory['units']) != set(self.names()):
            raise RuntimeError('installed/template unit inventory differs; review missing/extra units')
        policy=json.loads(self.config_policy.read_text()) if self.config_policy else {}
        if policy and policy.get('release')!=self.release: raise RuntimeError('config review is for another release')
        reviews=policy.get('reviews',{}).get(self.role,{})
        changes = {}
        units = {}
        folder = self.templates/('rpi' if self.role=='pi' else self.role)
        for name, old in inventory['units'].items():
            cwd = old['state'].get('WorkingDirectory')
            if cwd:
                if not Path(cwd).is_relative_to(self.root):
                    raise RuntimeError('unexpected existing source path')
                changes.update(review_config_files(cwd, self.source, reviews))
            for path, content in old['dropins'].items():
                if name != 'idcs-deepstream-video.service' or Path(path).name != KNOWN_OVERRIDE:
                    raise RuntimeError('unknown override must be explicitly reconciled: ' + path)
                expected = f'[Service]\nWorkingDirectory={cwd}\nEnvironment=PYTHONPATH={cwd}\n'
                if content != expected:
                    raise RuntimeError('known override contents changed')
            desired = render_unit((folder/name).read_text(), self.source)
            units[name] = dict(old=old, desired=desired, desired_sha256=digest(desired.encode()))
        # Canonicalize ignored model links to the external store; no tracked
        # source/config is mutated and no target data is deleted.
        links = []
        assets = self.source/'assets'
        for parent, dirs, files in os.walk(assets,followlinks=False):
            for name in dirs+files:
                path=Path(parent)/name
                if path.is_symlink():
                    target=path.resolve(strict=True)
                    links.append(dict(path=str(path),target=str(target),old=os.readlink(path)))
        self.save('plan.json',dict(role=self.role,source=str(self.source),commit=self.run(
            ['git','-C',self.source,'rev-parse','HEAD']).strip(),units=units,
            installer_sha256=digest(Path(__file__).read_bytes()),
            config_differences={k:v for k,v in changes.items() if v},asset_links=links,
            preserved_evidence='Old releases/trials and recordings retained; no deletion'))
        self.save('config-review.json',dict(policy_sha256=digest(self.config_policy.read_bytes()) if self.config_policy else None,
            reviews=reviews,config_differences={k:v for k,v in changes.items() if v}))
        return dict(units=len(units),source=str(self.source),config_changes={k:v for k,v in changes.items() if v})

    def validate(self):
        plan = self.load('plan.json')
        self.safety()
        native={}
        if self.role=='jetson':
            import yaml
            profile=yaml.safe_load((self.source/'configs/base/deepstream.yaml').read_text())['deepstream']['nvinfer_config']
            native=native_profile_assets(self.source,self.source/profile)
            library=native['custom-lib-path']['resolved']
            dependencies=self.run(['ldd',library])
            if 'not found' in dependencies: raise RuntimeError('unresolved parser dependency: '+dependencies)
            native['dependencies']=dependencies
        env = dict(os.environ,PYTHONPATH=str(self.source),PYTHONDONTWRITEBYTECODE='1')
        tests = ['tests/test_operator_agent.py','tests/test_pc_mode.py','tests/test_target_authority.py',
            'tests/test_panel_indicators.py','tests/test_rpi_gpio_config.py','tests/test_operator_menu.py',
            'tests/test_panel_hud.py','tests/test_tracking_reticle.py','tests/test_deepstream_metadata_adapter.py',
            'tests/test_deepstream_accepted_osd.py','tests/test_flight_recorder.py','tests/test_sim_truth_capture.py',
            'tests/test_ui_style.py','tests/test_hud_design.py','tests/test_video_health.py','tests/test_v2_ui_hud.py','tests/test_v2_hud_attitude.py',
            'tests/test_serial_io_ipc.py','tests/test_operator_display.py','tests/test_sim_score.py']
        tests = [p for p in tests if (self.source/p).exists()]
        if self.role=='pc':
            # Pi operator_display imports GstVideo, unavailable in the PC's
            # Python3.14 GI installation. This suite runs on Pi AND Jetson.
            tests=[p for p in tests if p!='tests/test_operator_menu.py']
        runner = ('import sys,types; p=types.ModuleType("tests"); p.__path__=['+repr(str(self.source/'tests'))+
            '];sys.modules["tests"]=p;import pytest;sys.exit(pytest.main(sys.argv[1:]))')
        # Pi's environment need not have pytest. Its supported unittest modules
        # cover Cairo and menu; safety/operator contract suites run on PC/Jetson.
        if self.role=='pi':
            output=self.run([self.python,'-m','unittest','-v','tests.test_panel_hud','tests.test_hud_design','tests.test_operator_menu',
                             'tests.test_rpi_gpio_config','tests.test_video_health','tests.test_sim_score'],cwd=self.source,env=env)
        else:
            output=self.run([self.python,'-c',runner,'-q','-p','no:cacheprovider',
                '--basetemp',self.evidence/f'test-tmp-{time.time_ns()}',*tests],cwd=self.source,env=env)
        (self.evidence/'tests.txt').write_text(output)
        checks=[]
        for name,item in plan['units'].items():
            for line in item['desired'].splitlines():
                if name not in PHYSICAL and line.startswith('ExecStartPre=') and ' --check' in line:
                    import shlex
                    args=shlex.split(line.split('=',1)[1])
                    # Check-only CLIs, never serial devices or process owners.
                    checks.append(dict(unit=name,command=args,output=self.run(args,cwd=self.source,env=env)))
        rendered=self.evidence/'rendered'
        rendered.mkdir(exist_ok=True)
        for name,item in plan['units'].items(): (rendered/name).write_text(item['desired'])
        verification=self.run(['systemd-analyze']+([] if self.role=='jetson' else ['--user'])+
            ['verify',*[str(rendered/n) for n in plan['units']]])
        self.save('validation.json',dict(passed=True,commit=plan['commit'],checks=checks,
            tests=output,systemd_verify=verification,native_assets=native,unit_hashes={n:i['desired_sha256'] for n,i in plan['units'].items()}))
        return dict(passed=True,tests=output,check_count=len(checks))

    def install(self, path, text):
        path=Path(path)
        if path.parent != self.unitdir and path.parent.parent != self.unitdir:
            raise ValueError('unexpected unit destination')
        if self.role=='jetson':
            temp=self.evidence/('install-'+path.name)
            temp.write_text(text)
            self.run(['sudo','-n','install','-D','-m','644',temp,path])
        else:
            temp=path.with_name(path.name+'.managed-tmp')
            temp.write_text(text)
            temp.replace(path)

    def restart_previous(self, previous):
        active=[n for n,s in previous.items() if s.get('ActiveState')=='active' and n.endswith('.service')]
        if any(n in PHYSICAL for n in active): raise RuntimeError('refuse physical activation')
        # SIM target dependencies must recover together. No HIL/camera source
        # switch is inferred from a deployment request.
        if self.role=='pc' and previous.get('idcs-sim.target',{}).get('ActiveState')=='active':
            self.run(self.ctl+['start','idcs-sim.target'])
        elif self.role=='pc' and previous.get('idcs-sim-swarm.target',{}).get('ActiveState')=='active':
            self.run(self.ctl+['start','idcs-sim-swarm.target'])
        for name in active:
            if self.state(name)['ActiveState']!='active': self.run(self.ctl+['start',name])

    def stop_previous(self, previous):
        names=[n for n,s in previous.items() if s.get('ActiveState')=='active']
        if any(n in PHYSICAL for n in names): raise RuntimeError('refuse physical stop/start workflow')
        if names: self.run(self.ctl+['stop',*names])

    def reset_software_failures(self, previous):
        names=[n for n,s in previous.items() if s.get('ActiveState')=='active' and n.endswith('.service')]
        if any(n in PHYSICAL for n in names): raise RuntimeError('refuse physical reset workflow')
        # systemd may unload a stopped recorder with no restart state. Do not
        # reset unloaded/zero-state units; only known nonzero counters need it.
        names=[n for n in names if int(self.state(n).get('NRestarts') or 0)>0]
        if names: self.run(self.ctl+['reset-failed',*names])

    def activate(self):
        plan=self.load('plan.json'); validation=self.load('validation.json')
        if not validation['passed'] or validation.get('commit')!=plan['commit'] or validation['unit_hashes']!={n:i['desired_sha256'] for n,i in plan['units'].items()}:
            raise RuntimeError('validation does not match plan')
        self.safety()
        if self.run(['git','-C',self.source,'rev-parse','HEAD']).strip()!=plan['commit'] or self.run(
                ['git','-C',self.source,'status','--porcelain','--untracked-files=no']).strip():
            raise RuntimeError('release changed after validation')
        previous={n:self.state(n) for n in plan['units']}
        self.check_owners()
        for name,item in plan['units'].items():
            if previous[name]!=item['old']['state'] or Path(item['old']['main_path']).read_text()!=item['old']['main_text']:
                raise RuntimeError('owner or installed unit changed since inventory: '+name)
        self.prepare_activation_epoch()
        self.save('rollback.json',dict(previous=previous,units={n:i['old'] for n,i in plan['units'].items()}))
        try:
            self.stop_previous(previous)
            # Previous counters remain in rollback.json; start a new explicit
            # software activation epoch, not an ambiguous historical count.
            self.reset_software_failures(previous)
            for name,item in plan['units'].items():
                self.install(item['old']['main_path'],item['desired'])
                for path in item['old']['dropins']:
                    # ONLY this identified, backed-up override; never a directory.
                    self.run(['sudo','-n','rm','--',path])
            for link in plan['asset_links']:
                path=Path(link['path'])
                if os.readlink(path)!=link['old']: raise RuntimeError('asset link changed since plan')
                tmp=path.with_name(path.name+'.managed-tmp')
                tmp.symlink_to(link['target']); tmp.replace(path)
            self.run(self.ctl+['daemon-reload'])
            self.restart_previous(previous)
            self.save('activation.json',dict(previous=previous,source=str(self.source),commit=plan['commit'],timestamp_ns=time.time_ns()))
            return self.verify()
        except BaseException as exc:
            try:
                self.rollback()
            except BaseException as rollback_error:
                self.save('failure.json',dict(error=str(exc),rollback_error=str(rollback_error),rolled_back=False))
                raise exc from rollback_error
            self.save('failure.json',dict(error=str(exc),rolled_back=True))
            raise

    def prepare_activation_epoch(self):
        # rollback() already archives failed activation receipts. A successful
        # receipt must never be bypassed because an older failure file exists.
        if (self.evidence/'activation.json').exists():
            raise RuntimeError('release already activated; verify or choose a new release')
        failure=self.evidence/'failure.json'
        if failure.exists():
            if not self.load('failure.json').get('rolled_back'):
                raise RuntimeError('unresolved rollback failure; explicit recovery required')
            failure.replace(self.evidence/f'rolled-back-failure-{time.time_ns()}.json')

    def verify(self):
        plan=self.load('plan.json'); receipt=self.load('activation.json')
        if self.run(['git','-C',self.source,'rev-parse','HEAD']).strip()!=plan['commit']:
            raise RuntimeError('release commit changed since activation')
        for name,item in plan['units'].items():
            if digest(Path(item['old']['main_path']).read_text().encode())!=item['desired_sha256']:
                raise RuntimeError('installed unit changed since activation: '+name)
        deadline=time.monotonic()+60
        while True:
            states=self.states(); self.safety()
            pending=[]
            for n,s in states.items():
                old=receipt['previous'][n]
                if s['ActiveState']!=old['ActiveState']: pending.append(n+' state')
                if n.endswith('.service') and s.get('WorkingDirectory')!=str(self.source): pending.append(n+' configured source')
                if s.get('MainPID') not in (None,'0','') and s.get('process_cwd')!=str(self.source): pending.append(n+' process source')
                if s.get('NRestarts','0') not in ('0',''): raise RuntimeError('new automatic restart: '+n)
                if s.get('DropInPaths'): raise RuntimeError('unmanaged override: '+n)
            if not pending: break
            if time.monotonic()>=deadline: raise RuntimeError('readiness timeout: '+repr(pending))
            time.sleep(.5)
        # Prove owner stability and allow media/health sampling to warm up.
        time.sleep(10)
        after=self.states()
        if after!=states: raise RuntimeError('owner changed during stability canary')
        self.safety()
        if self.run(['git','-C',self.source,'status','--porcelain','--untracked-files=no']).strip():
            raise RuntimeError('tracked release source mutated')
        self.check_owners()
        deadline=time.monotonic()+45
        while True:
            try:
                health=self.health()
                break
            except (RuntimeError,FileNotFoundError) as exc:
                if self.states()!=after: raise RuntimeError('owner changed while waiting for media health') from exc
                if time.monotonic()>=deadline: raise
                time.sleep(1)
        record=dict(role=self.role,release=self.release,commit=plan['commit'],source=str(self.source),
            units=after,verified_ns=time.time_ns(),old_evidence_retained=True,hardware_qualified=False,
            verification_installer_sha256=digest(Path(__file__).read_bytes()))
        record['health']=health
        self.save('verification.json',record)
        current=self.root/'deployments/managed'/f'{self.role}-current.json'
        current.write_text(json.dumps(record,indent=2)+'\n')
        return record

    def check_owners(self):
        expected={}
        for state in self.states().values():
            args=state.get('process_cmdline','').split()
            if '-m' in args: expected.setdefault(args[args.index('-m')+1],set()).add(int(state['MainPID']))
        owners={m:set() for m in expected}
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit(): continue
            try: args=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError): continue
            for i,a in enumerate(args[:-1]):
                if a=='-m' and args[i+1] in owners: owners[args[i+1]].add(int(entry.name))
        if owners!=expected or any(len(pids)!=1 for pids in owners.values()):
            raise RuntimeError(f'duplicate/unmanaged owners: expected={expected}, actual={owners}')
        return {m:sorted(p) for m,p in owners.items()}

    def health(self):
        states=self.states(); checks=[]
        if self.role=='jetson' and any(states[n]['ActiveState']=='active' for n in ('idcs-deepstream-video.service','idcs-deepstream-camera.service')):
            checks.append((Path('/run/idcs/deepstream-video-health.json'),lambda r:r['pipeline_fps']>=40 and r['return_fps']>=20))
        if self.role=='pc' and any(states[n]['ActiveState']=='active' for n in
                ('idcs-sim-controller.service','idcs-sim-swarm-controller.service')):
            # Target loss, coast and selection handover are valid in a dynamic
            # Swarm scene. Media health does not qualify detector effectiveness.
            checks.append((Path('/run/user/1000/idcs-sim/controller-health.json'),lambda r:r['ticks']>100 and r['snapshots']>20 and r['last_reason'] in ('tracking','manual','safety_hold','target_invalid','coasting','idle_return','target_switch_hold')))
        if self.role=='pi':
            checks.append((Path('/home/idcs/.local/state/idcs/operator-display.json'),lambda r:r['video_fps']>=25 and r['overlay_build_p95_ms']<=15 and 'tracking_mode' in r['hud_elements']))
        result={}
        for path,gate in checks:
            record=json.loads(path.read_text())
            if time.time()-path.stat().st_mtime>5 or not gate(record):
                raise RuntimeError('live media/control gate failed: '+str(path)+': '+repr(record))
            result[str(path)]=record
        return result

    def rollback(self):
        self.safety(); receipt=self.load('rollback.json')
        # Stop ONLY previous software owners, not physical services or a changed
        # source mode. Refuse if operator switched modes during the transaction.
        current=self.states()
        unexpected=[n for n,s in current.items() if s['ActiveState']=='active' and receipt['previous'][n]['ActiveState']!='active']
        if unexpected: raise RuntimeError('mode changed; refuse automatic rollback: '+repr(unexpected))
        self.stop_previous(receipt['previous'])
        for n,item in receipt['units'].items():
            self.install(item['main_path'],item['main_text'])
            for path,text in item['dropins'].items(): self.install(path,text)
        self.run(self.ctl+['daemon-reload']); self.restart_previous(receipt['previous']); self.safety()
        restored=self.states()
        self.save('rollback-result.json',dict(restored=restored,time_ns=time.time_ns()))
        self.invalidate_current_manifest(restored)
        activation=self.evidence/'activation.json'
        if activation.exists(): activation.replace(self.evidence/f'rolled-back-activation-{time.time_ns()}.json')
        return dict(restored=True)

    def invalidate_current_manifest(self, restored):
        current=self.root/'deployments/managed'/f'{self.role}-current.json'
        if current.exists():
            # It may still describe the older release if activation failed
            # before verify(). Retain it, but its PIDs are stale after recovery.
            current.replace(self.evidence/f'pre-rollback-current-{time.time_ns()}.json')
        current.write_text(json.dumps(dict(role=self.role,status='restored-unverified',
            units=restored,hardware_qualified=False,updated_ns=time.time_ns()),indent=2)+'\n')

    def audit(self):
        inventory=self.inventory()
        refs={}
        for n,item in inventory['units'].items():
            texts=[item['main_text'],*item['dropins'].values(),item['state'].get('process_cmdline',''),item['state'].get('process_cwd','')]
            for text in texts:
                for path in re.findall(r'/[^\s"\x27]+/IDCS-(?:trials|runtime|releases)[^\s"\x27]*',text): refs.setdefault(path,[]).append(n)
        report=dict(unit_and_process_references=refs,delete_permitted=False,
            reason='Evidence, old rollback plans and transitive asset/config/native references require separate review; this tool never deletes trial data.')
        self.save('reference-audit.json',report)
        return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('inventory','stage','plan','validate','activate','verify','rollback','audit'))
    ap.add_argument('--role',required=True,choices=('pc','jetson','pi'))
    ap.add_argument('--release',required=True)
    ap.add_argument('--root',type=Path,default=Path('/home/idcs/Desktop/project'))
    ap.add_argument('--templates',type=Path)
    ap.add_argument('--config-policy',type=Path,help='explicit full-config before/after semantic hash approvals')
    ap.add_argument('--bundle',type=Path); ap.add_argument('--commit')
    args=ap.parse_args(argv)
    manager=Deployment(args.root,args.role,args.release,args.templates,args.config_policy)
    result=manager.stage(args.bundle,args.commit) if args.phase=='stage' else getattr(manager,args.phase)()
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
