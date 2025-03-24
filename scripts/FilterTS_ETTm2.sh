if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ETTm2" ]; then
    mkdir ./logs/ETTm2
fi
export CUDA_VISIBLE_DEVICES=0

seq_len=96
model_name=FilterTS


pred_len=96
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --target 'OT' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --quantile 0.9 \
    --bandwidth 1 \
    --top_K_static_freqs 10 \
    --filter_type all \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1 >logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log

pred_len=192
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --target 'OT' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --quantile 0.9 \
    --bandwidth 1 \
    --top_K_static_freqs 10 \
    --filter_type all \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1 >logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log


pred_len=336
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --target 'OT' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --quantile 0.9 \
    --bandwidth 1 \
    --top_K_static_freqs 10 \
    --filter_type all \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --itr 1 >logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log


pred_len=720
python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --target 'OT' \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --quantile 0.9 \
    --bandwidth 1 \
    --top_K_static_freqs 10 \
    --filter_type all \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --itr 1 >logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log

