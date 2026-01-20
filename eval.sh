# /bin/bash
source /mnt/innovator/miniconda3/etc/profile.d/conda.sh
conda activate olympic
cd /mnt/innovator/code/wangcong/Evaluation/OlympicArena/code

export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN=$HF_TOKEN
export HF_DATASETS_CACHE="/mnt/innovator/data/wangcong/.cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL="1"

export OPENAI_API_KEY="sk-123456"
export OPENAI_BASE_URL="http://0.0.0.0:23333/v1"

MODEL_OUTPUT_DIR="/mnt/innovator/data/wangcong/data/eval/olympic"
rm -rf $MODEL_OUTPUT_DIR/*
# python ./main.py \
#     --model_output_dir $MODEL_OUTPUT_DIR \
#     --result_dir $MODEL_OUTPUT_DIR/results \
#     --model_path "/mnt/innovator/data/wangcong/model/Qwen3-30B-A3B-Base" \
#     --tp 8 \

python ./inference.py \
    --model_output_dir $MODEL_OUTPUT_DIR \
    --batch 2048 \
    --api_key $OPENAI_API_KEY \
    --base_url $OPENAI_BASE_URL \
    --model gpt-4o-text-only \
    --split test \
    --save_error

# python ./evaluation.py \
#     --model_output_dir $MODEL_OUTPUT_DIR \
#     --result_dir $MODEL_OUTPUT_DIR/results \
#     --model gpt-4o-text-only