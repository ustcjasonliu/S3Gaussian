#!/bin/bash


data_list=(
    "./data/waymo/processed/static32/003"
    "./data/waymo/processed/static32/019"
    "./data/waymo/processed/static32/036"
)


DATE=$(date '+%m%d')
output_root="./work_dirs/$DATE/static"
project=nvs100

for data_dir in "${data_list[@]}"; do
    # ��ȡ��Ŀ¼��basename
    model_name=$(basename "$data_dir")

    # ʹ��basename���޸�model_path
    model_path="$output_root/$project/$model_name"

    # ִ����ͬ�����ֻ�޸�-s��--model_path����
    CUDA_VISIBLE_DEVICES=$1 proxychains python train.py \
        -s "$data_dir" \
        --model_path "$model_path" \
        --expname 'waymo' \
        --configs "arguments/nvs.py" --eval_only

done
# bash scripts/static/run_nvs_list_2.sh 5