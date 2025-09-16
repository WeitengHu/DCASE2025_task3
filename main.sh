GPU_ID=7

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py  --ms --iv --gamma --itfm --exp "msic_itfm_synth_ss"

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py  --ms --iv --compfreq --exp "msi_fafs_synth_ss"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py  --exp "baseline_no_synth"

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py  --ms --iv --gamma --compfreq --exp "msic_fafs_synth_ss"

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py  --ms --iv --itfm --exp "msi_itfm_synth_ss"