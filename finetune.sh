GPU_ID=7

CUDA_VISIBLE_DEVICES=$GPU_ID
python3 main.py --exp "Finetune_msic_itfm" \
               --pretrained_exp "msic_itfm_synth_audio_multiACCDOA_20250907_122759" \
               --learning_rate 1e-4 \
               --weight_decay 1e-5 \
               --nb_epochs 20 \
               --itfm --ms --iv --gamma

               
