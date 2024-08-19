#!/bin/bash
data_list=(
    "./data/waymo/processed/dynamic32/016"
    "./data/waymo/processed/dynamic32/021"
    "./data/waymo/processed/dynamic32/022"
    "./data/waymo/processed/dynamic32/025"  
    "./data/waymo/processed/dynamic32/031"
    "./data/waymo/processed/dynamic32/034"
    "./data/waymo/processed/dynamic32/035" 
    "./data/waymo/processed/dynamic32/049"    
    "./data/waymo/processed/dynamic32/053"
    "./data/waymo/processed/dynamic32/080"
    "./data/waymo/processed/dynamic32/084"
    "./data/waymo/processed/dynamic32/086"
    "./data/waymo/processed/dynamic32/089"
    "./data/waymo/processed/dynamic32/094"  
    "./data/waymo/processed/dynamic32/096"
    "./data/waymo/processed/dynamic32/102"
    "./data/waymo/processed/dynamic32/111" 
    "./data/waymo/processed/dynamic32/222"    
    "./data/waymo/processed/dynamic32/323"
    "./data/waymo/processed/dynamic32/382"
    "./data/waymo/processed/dynamic32/402"
    "./data/waymo/processed/dynamic32/427"  
    "./data/waymo/processed/dynamic32/438"
    "./data/waymo/processed/dynamic32/546"
    "./data/waymo/processed/dynamic32/581" 
    "./data/waymo/processed/dynamic32/592"    
    "./data/waymo/processed/dynamic32/620"
    "./data/waymo/processed/dynamic32/640"
    "./data/waymo/processed/dynamic32/700" 
    "./data/waymo/processed/dynamic32/754"    
    "./data/waymo/processed/dynamic32/795"
    "./data/waymo/processed/dynamic32/796"
)


DATE=$(date '+%m%d%H%M')
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

done
# bash scripts/dynamic/run_recon_list_d.sh 2