GPU_ID=0

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "dcase2025_gcc" \
#                 --project 'task3_feature' \
#                 --gcc \
#                 --nb_epochs 50 \

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "dcase2025_slite" \
                --project 'task3_feature' \
                --slite \
                --nb_epochs 50 \