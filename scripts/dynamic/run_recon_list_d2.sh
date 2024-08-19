#!/bin/bash


data_list=(
    "./data/waymo/processed/dynamic32/016"
    "./data/waymo/processed/dynamic32/021"
)


DATE=$(date '+%m%d')
output_root="./work_dirs/$DATE/phase1/dynamic"
project=recon50

for data_dir in "${data_list[@]}"; do
    # 获取子目录的basename
    model_name=$(basename "$data_dir")


    # 使用basename来修改model_path
    model_path="$output_root/$project/$model_name"

    # 执行相同的命令，只修改-s和--model_path参数
    CUDA_VISIBLE_DEVICES=$1 proxychains python train.py \
        -s "$data_dir" \
        --model_path "$model_path" \
        --expname 'waymo' \
        # --start_checkpoint '/data1/hn/gaussianSim/gs4d/gs_1/work_dirs/0509/dynamic/reconstuction100/031/chkpnt_fine_10000.pth'
done
# bash scripts/dynamic/run_recon_list_d2.sh 1