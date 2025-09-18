GPU_ID=7

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py --exp "Finetune_baseline" \
               --pretrained_exp "baseline_no_synth_audio_multiACCDOA_20250916_201849" \
               --learning_rate 1e-4 \
               --weight_decay 1e-5 \
               --nb_epochs 20 \
               --finetune \
               --val_freq 1 \
               # --itfm --ms --iv --gamma


               
