# Stage 1 MuJoCo Sim2Sim

Run the converged pre-locomotion JIT policy in the plain G1 MuJoCo scene.

```bash
python deploy/deploy_mujoco_stage1/deploy_mujoco_stage1.py --policy model/policy.jit
```

Optional checks:

```bash
python deploy/deploy_mujoco_stage1/deploy_mujoco_stage1.py --policy model/policy.jit --dry-run
python deploy/deploy_mujoco_stage1/deploy_mujoco_stage1.py --policy model/policy.jit --cmd 0.3 0.0 0.0
```
