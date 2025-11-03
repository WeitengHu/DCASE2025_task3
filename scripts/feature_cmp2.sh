GPU_ID=1

CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "activate_dnorm" \
                --project 'task3_feature' \
                --dnorm \
                --nb_epochs 50 \

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "dcase2025_ms" \
#                 --project 'task3_feature' \
#                 --ms \
#                 --nb_epochs 50 \

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "dcase2025_iv" \
#                 --project 'task3_feature' \
#                 --iv \
#                 --nb_epochs 50 \

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "dcase2025_slite" \
#                 --project 'task3_feature' \
#                 --slite \
#                 --nb_epochs 50 \

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "dcase2025_ipd" \
#                 --project 'task3_feature' \
#                 --ipd \
#                 --nb_epochs 50 \

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "dcase2025_ild" \
#                 --project 'task3_feature' \
#                 --ild \
#                 --nb_epochs 50 \

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py  --exp "dcase2025_gcc" \
#                 --project 'task3_feature' \
#                 --gcc \
#                 --nb_epochs 50 \

