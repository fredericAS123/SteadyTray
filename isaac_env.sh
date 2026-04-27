"""
在云端服务器（不能使用docker）上安装 Isaac Sim 4.5.0 以及Isaac lab 的步骤： 
"""
# 1. 首先确保基础工具链完整，避免安装 flatdict 等底层包时崩溃。
cd /home/zjurobot/WTY/IsaacLab_SteadyTray
./isaaclab.sh --conda
conda activate env_isaaclab
python -m pip install --upgrade pip setuptools
python -m pip install flatdict # 可以no-isolate安装，避免安装过程中出现编码问题导致崩溃

# 2. 安装 Isaac Sim 4.5.0
# 先卸载所有已安装的 Isaac Sim 相关包，避免版本冲突
pip freeze | grep isaacsim | xargs pip uninstall -y
pip uninstall omniverse-kit -y
# 确保 pip 是最新版本
pip install --upgrade pip
# 安装 Isaac Sim 4.5.0 核心包及其缓存扩展
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# 设置环境变量接受最终用户许可协议 (EULA)
export OMNI_KIT_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=0
export CARB_GPU_P2P_TEST_ENABLED=0
# 为了一劳永逸，可以将此命令添加到你的 conda 环境激活脚本中
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 验证安装，单卡启动
isaacsim --/gpu/foundation/multiGpu/enabled=false --/gpu/foundation/deviceSelection="cuda:0"

# 3. 安装 Isaac Lab以及相关依赖(容器相关文件会让安装脚本误以为在容器中运行，导致安装失败，所以需要先重命名 .dockerenv 文件)
cd /home/zjurobot/WTY/IsaacLab_SteadyTray
sudo mv / .dockerenv /.dockerenv.bak && \
./isaaclab.sh --install && \
sudo mv /.dockerenv.bak /.dockerenv
