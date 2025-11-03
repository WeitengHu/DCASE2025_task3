GPU_ID=0



CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_iv" \
                --project 'task3_feature' \
                --iv \
                --nb_epochs 50 \


CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_gamma" \
                --project 'task3_feature' \
                --gamma \
                --nb_epochs 50 \


CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_ild" \
                --project 'task3_feature' \
                --ild \
                --nb_epochs 50 \
