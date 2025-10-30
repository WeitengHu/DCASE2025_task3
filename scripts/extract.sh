GPU_ID=0


CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/extract.py  --exp "extract" \
                --project 'task3_feature' \
                --nb_epochs 50 \
                --ipd \