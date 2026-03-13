#!/bin/bash

# ==========================================
# Usage: ./run_batch.sh [PLATFORM] [MODE]
# ==========================================

# 用户输入 -h 或 --help 打印帮助信息
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [PLATFORM] [MODE]"
    echo "  PLATFORM : NVIDIA (default), ILUVATAR, METAX, MOORE, HYGON"
    echo "  MODE     : run-all (default, full test), run-gpu (smart GPU test, skips data gen & CPU baseline)"
    echo "Examples:"
    echo "  $0                 # Run full test using NVIDIA"
    echo "  $0 HYGON           # Run full test using HYGON"
    echo "  $0 NVIDIA run-gpu  # Run only GPU test using NVIDIA (requires existing baseline data)"
    exit 0
fi

# 获取参数, 设置默认值
PLATFORM=${1:-"NVIDIA"}
MODE=${2:-"run-all"}

# 校验 MODE 参数是否合法
if [[ "$MODE" != "run-all" && "$MODE" != "run-gpu" ]]; then
    echo "❌ [Error] Unsupported test mode: $MODE. Must be 'run-all' or 'run-gpu'."
    exit 1
fi

# 全局固定参数
SEED=0
Q=100

# 定义 N 的取值列表
N_VALUES=(
    1000
    5000
    10000
    50000
    100000
    500000
)

# 日志文件
LOG_FILE="test_${PLATFORM}_${MODE}.log"

echo "==================================================="
echo "  🚀 Starting Tests"
echo "  PLATFORM : $PLATFORM"
echo "  MODE     : $MODE"
echo "  SEED     : $SEED"
echo "  Q        : $Q"
echo "  LOG FILE : $LOG_FILE"
echo "==================================================="

# 清空或创建日志文件, 并写入头部信息
echo "--- Batch tests started at $(date) ---" > "$LOG_FILE"
echo "Platform: $PLATFORM, Mode: $MODE" >> "$LOG_FILE"

# 循环执行测试
for N in "${N_VALUES[@]}"; do
    # 自动计算 M = 10 * N
    M=$((N * 10))

    echo ""
    echo "---------------------------------------------------"
    echo "  ⏳ Running: N=$N, M=$M"
    echo "---------------------------------------------------"

    # 调用 Makefile, 传入对应的目标 (MODE) 和参数
    # 2>&1 | tee -a 将标准输出和错误输出都追加到日志并打印在屏幕上
    make $MODE N="$N" M="$M" SEED="$SEED" Q="$Q" PLATFORM="$PLATFORM" 2>&1 | tee -a "$LOG_FILE"

    # 捕获 make 命令的真实退出状态码
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ [Error] An error occurred while running $MODE N=$N M=$M (e.g., OOM or Segmentation Fault)!" | tee -a "$LOG_FILE"
        echo "⛔ Terminating subsequent larger-scale tests. Please check the log for details." | tee -a "$LOG_FILE"
        exit 1
    fi
done

echo ""
echo "==================================================="
echo "  ✅ All test combinations completed!"
echo "  For detailed results, please check: $LOG_FILE"
echo "==================================================="