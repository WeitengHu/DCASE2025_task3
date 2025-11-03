GPU_ID=2

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_ms" \
                --project 'task3_feature' \
                --ms \
                --nb_epochs 50 \

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_gcc" \
                --project 'task3_feature' \
                --gcc \
                --nb_epochs 50 \

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_slite" \
                --project 'task3_feature' \
                --slite \
                --nb_epochs 50 \

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_ild" \
                --project 'task3_feature' \
                --ild \
                --nb_epochs 50 \