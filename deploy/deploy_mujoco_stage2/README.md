# Stage 2 MuJoCo Sim2Sim

Run a tray-holding JIT policy with the tray/plate scene. The config removes the object body at load time so Stage 2 does not depend on object observations.

```bash
python deploy/deploy_mujoco_stage2/deploy_mujoco_stage2.py --policy <stage2_policy_jit>
```

Optional checks:

```bash
python deploy/deploy_mujoco_stage2/deploy_mujoco_stage2.py --policy <stage2_policy_jit> --dry-run
python deploy/deploy_mujoco_stage2/deploy_mujoco_stage2.py --policy <stage2_policy_jit> --cmd 0.3 0.0 0.0
```
