# 该文件路径
This_path=SemEval2024/
# wandb的项目名称
project_name=semeval
#每次实验的名称
Run_name=Abel-sft-1-noCND

# 预训练模型的位置
PreTrained_model=Abel002
# train set
Data_Path=SemEval2024/1_sft_data_generated_noCND.json
# validation set
Valid_path=
# 若使用validation，则需要添加一下参数：(参考trainer文档)
    # --valid_data_path $Valid_path \
    # --do_eval \
    # --eval_steps 400 \
    # --evaluation_strategy "steps" \

# output pat
Output_dir=$This_path/ISFT-output/$Run_name
# deepspeed config path

ds_config_stage2=$This_path/configs/ds_config_stage2.json
ds_config_stage3=$This_path/configs/default_offload_opt_param.json

# 可视化
# report_to="wandb"

# 不能一边train一边evaluate

num_train_epochs=1

Train_File=$This_path/sft_train.py
# 注意修改GPUs时，也要指定nproc_per_node的值！！！！
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --nnodes=1 --master_port=9901 $Train_File \

# 用6，7的时候bf16和t2都要false
deepspeed --include localhost:2,3 --master_port=9902 $Train_File \
    --model_name_or_path $PreTrained_model \
    --do_train \
    --train_data_path $Data_Path \
    --bf16 True \
    --output_dir $Output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --num_train_epochs $num_train_epochs \
    --run_name  $Run_name    \
    --model_max_length  1024 \
    --deepspeed $ds_config_stage2 \
    # --valid_data_path $Valid_path \
    # --do_eval \
    # --eval_steps 400 \
    # --project_name $project_name \
    # --report_to $report_to \
    # --fsdp "full_shard auto_wrap offload" \
    # --fsdp_transformer_layer_cls_to_wrap "GPTNeoBlock" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    # --learning_rate 2e-5 \
    
