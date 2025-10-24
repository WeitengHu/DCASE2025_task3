GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py --exp "Finetune_seld_dnorm" \
               --pretrained_exp "seldnet_audio_multiACCDOA_20251023_184819" \
               --learning_rate 1e-4 \
               --weight_decay 1e-5 \
               --nb_epochs 20 \
               --finetune \
               --val_freq 1 \
               # --itfm --ms --iv --gamma


               
