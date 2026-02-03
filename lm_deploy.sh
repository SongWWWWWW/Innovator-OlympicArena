# /bin/bash

source /mnt/innovator/miniconda3/etc/profile.d/conda.sh
conda activate scievalkit

export model_path="/mnt/innovator/data/wangcong/model/Olmo-3-7B-Instruct"
export model_name="gpt-4o-text-only"


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lmdeploy serve api_server $model_path \
#     --model-name $model_name  \
#     --server-port 23333 \
#     --tp 8 \
#     --max-batch-size 2048 \
#     --session-len 8192 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --served-model-name $model_name \
    --port 23333 \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --max-num-seqs 2048 \
    --trust-remote-code
    

