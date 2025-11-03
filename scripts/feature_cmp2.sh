GPU_ID=1


CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_stpacc" \
                --project 'task3_feature' \
                --stpacc \
                --nb_epochs 50 \

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_slite" \
                --project 'task3_feature' \
                --slite \
                --nb_epochs 50 \



