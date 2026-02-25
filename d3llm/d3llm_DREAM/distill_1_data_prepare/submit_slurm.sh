#!/bin/bash

# 配置参数
START=0          # 起始索引
END=100000       # 结束索引（根据你的 --max_data_num 参数）
STEP=1000        # 每个任务处理的数据量
SCRIPT_PATH="d3llm/d3llm_DREAM/distill_1_data_prepare/d3llm_dream_generate_partly.py"

# 优先尝试的分区顺序
PARTITIONS=("a100" "3090")

# 创建日志目录
mkdir -p slurm_logs
mkdir -p generated_data

# 循环提交任务
for ((start_idx=$START; start_idx<$END; start_idx+=$STEP)); do
    end_idx=$((start_idx + STEP))
    
    # 确保不超过总范围
    if [ $end_idx -gt $END ]; then
        end_idx=$END
    fi
    
    # 生成输出文件名
    output_file="generated_data/trajectory_data_${start_idx}_${end_idx}.json"
    
    echo "提交任务: 处理数据 [$start_idx, $end_idx) -> $output_file"
    
    # 尝试在不同分区提交任务
    job_submitted=false
    
    for partition in "${PARTITIONS[@]}"; do
        # 检查分区是否存在
        if sinfo -p $partition 2>/dev/null | grep -q $partition; then
            # 提交任务
            sbatch --partition=$partition \
                   --job-name="dream_gen_${start_idx}_${end_idx}" \
                   --output="slurm_logs/dream_gen_${start_idx}_${end_idx}_%j.out" \
                   --error="slurm_logs/dream_gen_${start_idx}_${end_idx}_%j.err" \
               << 'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00


source /mnt/fast/nobackup/users/mc03002/miniconda3/bin/activate
conda activate distill

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Processing indices: $start_idx to $end_idx"

# 运行 Python 脚本
python d3llm_dream_generate_partly.py \
    --start_idx $start_idx \
    --end_idx $end_idx \
    --steps 256 \
    --gen_length 256 \
    --block_length 32 \
    --output_file $output_file \
    --max_data_num 100000

# 检查任务是否成功
if [ $? -eq 0 ]; then
    echo "任务成功完成于: $(date)"
else
    echo "任务失败于: $(date)" >&2
    exit 1
fi
EOF
            job_submitted=true
            break
        else
            echo "分区 $partition 不可用，尝试下一个..."
        fi
    done
    
    if [ "$job_submitted" = false ]; then
        echo "错误: 没有可用的分区 (a100, 3090) 来提交任务"
        exit 1
    fi
    
    # 稍微等待一下，避免提交太快
    sleep 1
done

echo "所有任务提交完成！"
echo "检查任务状态: squeue -u $USER"