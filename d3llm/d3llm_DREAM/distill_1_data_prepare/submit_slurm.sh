#!/bin/bash

# 配置参数
START=96000
# END=122000
END=112000
STEP=2000
SCRIPT_PATH="d3llm/d3llm_DREAM/distill_1_data_prepare/d3llm_dream_generate_partly.py"

mkdir -p slurm_logs
mkdir -p generated_data

for ((start_idx=$START; start_idx<$END; start_idx+=$STEP)); do
    end_idx=$((start_idx + STEP))
    
    if [ $end_idx -gt $END ]; then
        end_idx=$END
    fi
    
    output_file="generated_data/trajectory_data_${start_idx}_${end_idx}.json"
    
    echo "提交任务: 处理数据 [$start_idx, $end_idx) -> $output_file"
    
    sbatch --job-name="dream_gen_${start_idx}_${end_idx}" \
            --partition=a100 \
            --output="slurm_logs/dream_gen_${start_idx}_${end_idx}_%j.out" \
            --error="slurm_logs/dream_gen_${start_idx}_${end_idx}_%j.err" \
    << EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00

source /mnt/fast/nobackup/users/mc03002/miniconda3/bin/activate
conda activate distill

echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODENAME"
echo "Partition: \$SLURM_JOB_PARTITION"
echo "Processing indices: $start_idx to $end_idx"

python $SCRIPT_PATH \\
    --start_idx $start_idx \\
    --end_idx $end_idx \\
    --steps 512 \\
    --gen_length 512 \\
    --block_length 1 \\
    --output_file $output_file \\
    --max_data_num 100000

if [ \$? -eq 0 ]; then
    echo "任务成功完成于: \$(date)"
else
    echo "任务失败于: \$(date)" >&2
    exit 1
fi
EOF
    
    sleep 1
done

echo "所有任务提交完成！"
echo "检查任务状态: squeue -u $USER"