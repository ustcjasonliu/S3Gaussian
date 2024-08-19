#!/bin/bash


data_list=(
    "./data/waymo/processed/dynamic32/016"
    "./data/waymo/processed/dynamic32/021"  
    "./data/waymo/processed/dynamic32/022"
)


DATE=$(date '+%m%d')
output_root="./work_dirs/08181607/phase2/dynamic"
project=recon50

for data_dir in "${data_list[@]}"; do
    # 获取子目录的basename
    model_name=$(basename "$data_dir")
    echo "model name $model_name"

    # 使用basename来修改model_path
    model_path="$output_root/$project/$model_name"

    echo "model path $model_path"

    # 执行相同的命令，只修改-s和--model_path参数
    CUDA_VISIBLE_DEVICES=$1 proxychains python train.py \
        -s "$data_dir" \
        --model_path "$model_path" \
        --expname 'waymo' \
        --configs "arguments/stage2.py" \
        --prior_checkpoint "/mnt/3dvision-cpfs/jason/S3Gaussian/work_dirs/08181607/phase1/dynamic/recon50/$model_name/chkpnt_fine_50000.pth"

done
# bash scripts/stage2/dynamic/run_recon_list_d.sh 3