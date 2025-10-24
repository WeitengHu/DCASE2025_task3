GPU_ID=2

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py --exp "Finetune_baseline" \
               --pretrained_exp "baseline_audio_multiACCDOA_20251023_144536" \
               --learning_rate 1e-4 \
               --weight_decay 1e-5 \
               --nb_epochs 20 \
               --finetune \
               --val_freq 1 \
               --dnorm \
               # --itfm --ms --iv --gamma