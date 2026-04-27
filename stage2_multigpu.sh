# 1. 开启了 IOMMU 的情况下运行了 Isaac Sim 底层的 P2P 带宽和延迟验证（p2pBandwidthLatencyTest.cu）。由于 4090 在某些主板和驱动环境下缺乏官方的 NVLink/P2P 内存共享支持，这个验证步骤经常会爆显存警告
# 多卡同步时出现通信卡死，可以尝试在环境变量中禁用 P2P
export NCCL_P2P_DISABLE=1

# 2. 创建软链接，指向正确的log路径
ln -s /home/zjurobot/WTY/SteadyTray/logs/rsl_rl/g1_steady_tray_pre_locomotion/2026-04-23_05-43-35_pretrain_loco_multigpu /home/zjurobot/WTY/SteadyTray/logs/rsl_rl/g1_steady_tray/stage1_weights

# 3. 启动多卡训练(2)
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/rsl_rl/train.py \
    --task G1-Steady-Tray \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run stage1_weights \
    --distributed \
    --run_name "tray_finetune"

# 备用指令（如有报错可尝试）
# export CUDA_VISIBLE_DEVICES=0,1
# 清理僵尸进程（可ai）
# ps -ef | grep python | grep SteadyTray | awk '{print $2}' | xargs kill -9

# 4. 观察显卡使用情况(1s刷新一次)
watch -n 1 nvidia-smi

# 5. 观察训练曲线(需用端口映射到本地查看)

tensorboard --logdir /home/zjurobot/WTY/SteadyTray/logs/rsl_rl/g1_steady_tray/2026-04-24_01-23-29_tray_finetune \
    --port 6006 \
    # --bind_all (如允许公网访问，则添加此参数。只需http://服务器IP:6006访问即可)
ssh -L 6006:localhost:6006 zjurobot@ccea30caeb46 # 连接服务器后，在本地浏览器访问 http://localhost:6006 即可查看训练曲线
# 有时候vscode会直接提示转发打开浏览器，直接点击即可访问 http://localhost:6006