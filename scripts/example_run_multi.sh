#!/bin/bash

# Example script for running distributed reward model inference with different model types


# Example 8: Model Parallelism for 70B models (use run_rm_host_70b.py)
echo "Example 8: Model Parallelism for 70B models..."
python scripts/run_rm_host_70b.py \
    --model_name_or_path "allenai/Llama-3.1-70B-Instruct-RM-RB2" \
    --model_name "allenai/Llama-3.1-70B-Instruct-RM-RB2" \
    --output_path "results/llama_70b_scores" \
    --max_length 2048 \
    --torch_dtype "bfloat16" \
    --device_map "auto"

# Example 9: Model Parallelism with custom device mapping
echo "Example 9: Model Parallelism with balanced device mapping..."
python scripts/run_rm_host_70b.py \
    --model_name_or_path "infly/INF-ORM-Llama3.1-70B" \
    --model_name "infly/INF-ORM-Llama3.1-70B" \
    --output_path "results/inf_orm_70b_scores" \
    --max_length 2048 \
    --torch_dtype "bfloat16" \
    --device_map "balanced"

echo "Done! Check the results in the output directories." 