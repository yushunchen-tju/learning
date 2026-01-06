export CUDA_VISIBLE_DEVICES=0
CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="ocean/${CURRENT_TIME}_PredFormer_depth4_FacTS_sd0.25_dp0.1_ps4_bs8_256_8_32_5e-4_Adamw_cosine_200ep"

python tools/train.py \
    --config_file configs/ocean/PredFormer.py \
    --dataname ocean \
    --res_dir work_dirs \
    --batch_size 8 \
    --epoch 50 \
    --overwrite \
    --lr 5e-4 \
    --sched cosine \
    --warmup_epoch 0 \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME"  \
    --tb_dir logs_tb/03_08