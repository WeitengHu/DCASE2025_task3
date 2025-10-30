GPU_ID=5


CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "Finetune_mini_dnorm" \
                --project 'task3_feature' \
                --pretrained_exp "mini_dnorm_audio_multiACCDOA_20251027_155303" \
                --learning_rate 1e-4 \
                --weight_decay 1e-5 \
                --nb_epochs 20 \
                --finetune \
                --val_freq 1 \
                --dnorm \
