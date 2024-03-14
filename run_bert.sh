CUDA_VISIBLE_DEVICES=0,1 python train_bert.py --output_dir='./bert_output4/' \
 --do_train \
 --learning_rate=2e-5 \
 --per_device_train_batch_size=32 \
 --gradient_accumulation_steps=32 \
 --warmup_ratio=0.05\
 --num_train_epochs=10.0\
 --do_eval\
 --evaluation_strategy='epoch' \
 --per_device_eval_batch_size=32 \
 --save_strategy="epoch"\
 --do_predict
 

