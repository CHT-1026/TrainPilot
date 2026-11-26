"""以替身Python检查V6真实Bash入口；不启动服务器训练。"""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
GIT=shutil.which('git')
GIT_BASH=Path(GIT).resolve().parents[1]/'bin/bash.exe' if GIT else None
BASH=str(GIT_BASH) if os.name=='nt' and GIT_BASH and GIT_BASH.exists() else shutil.which('bash')


@unittest.skipUnless(BASH,'需要Bash检查服务器入口')
class V6LauncherChecks(unittest.TestCase):
    def launch(self,root,mode,resume=False,locked=False):
        root=Path(root);run=root/'run';run.mkdir()
        if resume:(run/'checkpoint.pt').touch()
        arguments=root/'args.txt'
        env={**os.environ,'TRAINPILOT_SFT_MODEL':'/models/full-sft','TRAINPILOT_V5_ADAPTER':'/adapters/v5',
             'TRAINPILOT_LORA_RUN':run.as_posix(),'LAUNCH_ARGS':arguments.as_posix(),
             'SCRIPT':(ROOT/'model/trainer/run_trainpilot_lora_v6.sh').as_posix(),'MODE':mode,
             'LOCKED':'1' if locked else '0'}
        command='python() { printf "%s\\n" "$@" > "$LAUNCH_ARGS"; }; flock() { return "$LOCKED"; }; export -f python flock; bash "$SCRIPT" "$MODE"'
        process=subprocess.run([BASH,'-c',command],env=env,cwd=ROOT,capture_output=True,encoding='utf-8',errors='replace',timeout=30)
        return process,arguments.read_text(encoding='utf-8').splitlines() if arguments.exists() else []

    def test_smoke_loads_v5_and_limits_ten_updates(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:process,args=self.launch(root,'smoke')
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertEqual(args[args.index('--init-adapter')+1],'/adapters/v5')
        self.assertEqual(args[args.index('--model')+1],'/models/full-sft')
        self.assertEqual(args[args.index('--config')+1],'model/config/lora_v6.json')
        self.assertEqual(args[args.index('--max-updates')+1],'10')
        self.assertNotIn('--resume',args)

    def test_train_resumes_own_v6_only(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:
            process,args=self.launch(root,'train',resume=True)
            self.assertEqual(args[args.index('--resume')+1],(Path(root)/'run/checkpoint.pt').as_posix())
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertNotIn('--max-updates',args)
        self.assertEqual(args[args.index('--init-adapter')+1],'/adapters/v5')

    def test_check_only_and_duplicate_training_lock(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:process,args=self.launch(root,'check')
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertIn('--prepare-only',args)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:process,args=self.launch(root,'train',locked=True)
        self.assertNotEqual(process.returncode,0)
        self.assertEqual(args,[])

    def test_default_plan_is_one_epoch_130_updates(self):
        config=json.loads((ROOT/'model/config/lora_v6.json').read_text(encoding='utf-8'))
        manifest=json.loads((ROOT/'model/dataset/datasets/trainpilot_agent/v6/manifest.json').read_text(encoding='utf-8'))
        updates=math.ceil(manifest['outputs']['train']['records']/config['batch_size']/config['accumulation_steps'])*config['epochs']
        self.assertEqual(config['epochs'],1)
        self.assertEqual(config['learning_rate'],1e-5)
        self.assertEqual(updates,130)


if __name__=='__main__':unittest.main()
