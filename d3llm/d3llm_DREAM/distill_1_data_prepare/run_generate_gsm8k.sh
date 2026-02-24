

start_idx=0
end_idx=1000
python d3llm_dream_generate_partly.py \
        --start_idx $start_idx \
        --end_idx $end_idx \
        --steps 256 \
        --gen_length 256 \
        --block_length 32 \
        --output_file trajectory_data.json \
        --max_data_num 10000