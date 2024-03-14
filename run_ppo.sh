# 该文件路径
This_path=SemEval2024/
#每次实验的名称
Run_name=Abel-sft2-ppo-output

# output pat
Output_dir=$This_path/output/$Run_name
# deepspeed config path

ds_config_stage2=configs/ds_config_stage2.json
ds_config_stage3=configs/ds_config_stage3.json


Train_File=$This_path/ppo_trl.py

# accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml \
#     --num_processes {NUM_GPUS} path_to_script.py \

# accelerate launch --config_file $ds_config_stage3

deepspeed --include localhost:0,1,2,3 --master_port=9902 $Train_File \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing_kwargs {"use_reentrant":False} \
    --gradient_checkpointing False \
    --use_cache False \
    --use_reentrant False\
    --num_train_epochs 5 \
    --model_max_length  512 \
    --deepspeed $ds_config_stage3 \
    --use_peft \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 300 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
