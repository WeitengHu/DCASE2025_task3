GPU_ID=2


CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_ipd" \
                --project 'task3_feature' \
                --ipd \
                --nb_epochs 50 \

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_gcc" \
                --project 'task3_feature' \
                --gcc \
                --nb_epochs 50 \

