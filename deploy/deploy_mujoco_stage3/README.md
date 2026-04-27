# Stage 3 MuJoCo Sim2Sim

Run an exported Stage 3 adapter teacher JIT policy with tray plus object observations.

Export a Stage 3 checkpoint first:

```bash
python deploy/scripts/batch_processing.py \
  --input_path <stage3_model_pt> \
  --output_path <output_dir>
```

Then run:

```bash
python deploy/deploy_mujoco_stage3/deploy_mujoco_stage3.py --policy <output_dir>/exported/<policy_jit>
```

Optional check:

```bash
python deploy/deploy_mujoco_stage3/deploy_mujoco_stage3.py --policy <stage3_teacher_jit> --dry-run
```
