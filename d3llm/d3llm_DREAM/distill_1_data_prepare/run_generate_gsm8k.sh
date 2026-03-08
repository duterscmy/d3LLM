

start_idx=0
end_idx=30
python d3llm/d3llm_DREAM/distill_1_data_prepare/d3llm_dream_generate_partly.py \
        --start_idx $start_idx \
        --end_idx $end_idx \
        --steps 512 \
        --gen_length 512 --block_length 32 \
        --output_file trajectory_data_ar.json \
        --max_data_num 100000