#!/bin/bash
echo "🚀 开始配置 SteadyTray 运行环境..."

# 1. 更新并安装系统工具
<<<<<<< HEAD
apt-get update
apt-get install -y git-lfs tmux

# 2. 安装 Python 拓展包
python -m pip install -e /workspace/steadytray/source/steadytray

# 3. 检查lfs文件
git config --global --add safe.directory /workspace/steadytray
git lfs install
git lfs pull
=======
sudo apt-get update
sudo apt-get install -y git-lfs tmux

# 2. 安装 Python 拓展包
python -m pip install -e source/steadytray

# 3. 检查lfs文件
# git config --global --add safe.directory /workspace/steadytray
# git lfs install
# git lfs pull
>>>>>>> 7b55b92c4a704cf4a9fa3aa0acf7196c7e999692

echo "✅ 环境配置完成！可以开始了。"
