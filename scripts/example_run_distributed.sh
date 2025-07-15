#!/bin/bash

# Example script for running distributed reward model inference with different model types

# Set environment variables
export HF_TOKEN="your_huggingface_token_here"  # Optional, for private models
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Set available GPUs

# Example 1: INF-ORM-Llama3.1-70B
echo "Example 1: Running with INF-ORM-Llama3.1-70B..."
accelerate launch --config_file config/accelerate_config.yaml \
    scripts/run_rm_distributed.py \
    --model_name_or_path "infly/INF-ORM-Llama3.1-70B" \
    --model_name "infly/INF-ORM-Llama3.1-70B" \
    --output_path "results/inf_orm_scores" \
    --max_length 2048 \
    --torch_dtype "bfloat16"

# Example 2: RM-Gemma-7B (pipeline model)
echo "Example 2: Running with RM-Gemma-7B (pipeline)..."
accelerate launch --config_file config/accelerate_config.yaml \
    scripts/run_rm_distributed.py \
    --model_name_or_path "weqweasdas/RM-Gemma-7B" \
    --model_name "weqweasdas/RM-Gemma-7B" \
    --output_path "results/rm_gemma_scores" \
    --max_length 2048 \
    --torch_dtype "bfloat16"

# Example 3: Skywork-Reward-Gemma-2-27B
echo "Example 3: Running with Skywork-Reward-Gemma-2-27B..."
accelerate launch --config_file config/accelerate_config.yaml \
    scripts/run_rm_distributed.py \
    --model_name_or_path "Skywork/Skywork-Reward-Gemma-2-27B" \
    --model_name "Skywork/Skywork-Reward-Gemma-2-27B" \
    --output_path "results/skywork_scores" \
    --max_length 2048 \
    --torch_dtype "bfloat16"

# Example 4: QRM-Gemma-2-27B (custom output format)
echo "Example 4: Running with QRM-Gemma-2-27B..."
accelerate launch --config_file config/accelerate_config.yaml \
    scripts/run_rm_distributed.py \
    --model_name_or_path "nicolinho/QRM-Gemma-2-27B" \
    --model_name "nicolinho/QRM-Gemma-2-27B" \
    --output_path "results/qrm_scores" \
    --max_length 2048 \
    --torch_dtype "bfloat16"

# Example 5: URM-LLaMa-3.1-8B (requires attention mask)
echo "Example 5: Running with URM-LLaMa-3.1-8B..."
accelerate launch --config_file config/accelerate_config.yaml \
    scripts/run_rm_distributed.py \
    --model_name_or_path "LxzGordon/URM-LLaMa-3.1-8B" \
    --model_name "LxzGordon/URM-LLaMa-3.1-8B" \
    --output_path "results/urm_scores" \
    --max_length 2048 \
    --torch_dtype "bfloat16"

# Example 6: Llama-3.1-70B-Instruct-RM-RB2 (standard model)
echo "Example 6: Running with Llama-3.1-70B-Instruct-RM-RB2..."
accelerate launch --config_file config/accelerate_config.yaml \
    scripts/run_rm_distributed.py \
    --model_name_or_path "allenai/Llama-3.1-70B-Instruct-RM-RB2" \
    --model_name "allenai/Llama-3.1-70B-Instruct-RM-RB2" \
    --output_path "results/llama_rm_scores" \
    --max_length 2048 \
    --torch_dtype "bfloat16"

# Example 7: Debug mode with any model
echo "Example 7: Debug mode with Skywork model..."
accelerate launch --config_file config/accelerate_config.yaml \
    scripts/run_rm_distributed.py \
    --model_name_or_path "Skywork/Skywork-Reward-Gemma-2-27B" \
    --model_name "Skywork/Skywork-Reward-Gemma-2-27B" \
    --output_path "results/debug_scores" \
    --debug \
    --torch_dtype "bfloat16"

echo "Done! Check the results in the output directories." 