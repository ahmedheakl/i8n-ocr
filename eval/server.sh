#!/usr/bin/bash
ENV=".ocr-eval"
source "/home/$USER/miniconda3/etc/profile.d/conda.sh"
conda activate $ENV
export PATH=~/local_cuda/bin:$PATH
export LD_LIBRARY_PATH=~/local_cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2
export API_PORT=8079
llamafactory-cli  api yamls/qwen2_5_7b.yaml