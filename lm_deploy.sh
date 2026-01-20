# /bin/bash

source /mnt/innovator/miniconda3/etc/profile.d/conda.sh
conda activate scievalkit

export model_path="/mnt/innovator/data/wangcong/model/cpt2hf/qwen3-8b-base2"
export model_name="gpt-4o-text-only"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lmdeploy serve api_server $model_path \
    --model-name $model_name  \
    --server-port 23333 \
    --tp 8 \
    --max-batch-size 2048 \
    --session-len 8192 \