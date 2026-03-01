#!/bin/bash
#SBATCH --job-name="distill"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                # 请求8块GPU
#SBATCH --time=2:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err



### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate distill

cd /projects/u6er/public/mingyu/d3LLM/utils/utils_Dream/eval_instruct

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args torch_compile=False,pretrained=/lus/lfs1aip2/projects/public/u6er/mingyu/models/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks gsm8k_cot_zeroshot \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path eval_tmp/gsm8k_cot_zeroshot_dream \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template