#!/bin/bash

# 同步本地代码到远程 WSL2 机器
# 使用方法: ./sync_to_remote.sh

# 配置参数
REMOTE_HOST="wsl2"
REMOTE_USER="feynmanzhao"
REMOTE_PORT="22"
REMOTE_DIR="~/nanochat"  # 远程目标目录
LOCAL_DIR="/Users/zzy/CodeBuddy/nanochat"  # 本地源目录

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}开始同步代码到远程机器${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查本地目录是否存在
if [ ! -d "$LOCAL_DIR" ]; then
    echo -e "${RED}错误: 本地目录 $LOCAL_DIR 不存在${NC}"
    exit 1
fi

# 显示同步信息
echo -e "${YELLOW}本地目录:${NC} $LOCAL_DIR"
echo -e "${YELLOW}远程主机:${NC} $REMOTE_HOST"
echo -e "${YELLOW}远程目录:${NC} $REMOTE_DIR"
echo -e "${YELLOW}端口:${NC} $REMOTE_PORT"
echo ""

# 同步前先测试连接
echo -e "${YELLOW}测试 SSH 连接...${NC}"
ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "echo '连接成功'" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 无法连接到远程主机 $REMOTE_HOST${NC}"
    echo -e "${YELLOW}请确保:${NC}"
    echo "  1. SSH 密钥已配置或能正常输入密码"
    echo "  2. 网络连接正常"
    echo "  3. 远程主机 IP: 192.168.1.90 端口: 22 可访问"
    exit 1
fi
echo -e "${GREEN}SSH 连接测试通过${NC}"
echo ""

# 创建远程目录（如果不存在）
echo -e "${YELLOW}检查并创建远程目录...${NC}"
ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"
echo -e "${GREEN}远程目录准备完成${NC}"
echo ""

# 执行同步
echo -e "${YELLOW}开始同步文件...${NC}"
echo -e "${YELLOW}排除的文件/目录:${NC}"
echo "  - .git/"
echo "  - __pycache__/"
echo "  - .DS_Store"
echo "  - *.pyc"
echo "  - .pytest_cache/"
echo "  - .codebuddy/"
echo "  - node_modules/"
echo ""

rsync -avz --delete \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='.codebuddy/' \
    --exclude='node_modules/' \
    --exclude='*.egg-info/' \
    --exclude='.tox/' \
    --exclude='.coverage' \
    --exclude='.mypy_cache/' \
    -e "ssh -p $REMOTE_PORT" \
    "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

# 检查同步结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}同步成功完成!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}现在可以执行以下命令连接到远程机器:${NC}"
    echo "  ssh wsl2"
    echo ""
    echo -e "${YELLOW}然后在远程机器上执行:${NC}"
    echo "  cd ~/nanochat"
    echo "  # 你的其他命令..."
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}同步失败!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
