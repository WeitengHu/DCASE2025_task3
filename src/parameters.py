"""
parameters.py

This module stores all the configurable parameters and hyperparameters used
across the project, ensuring easy tuning and reproducibility.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""

params = {

    # choose task
    'modality': 'audio',  # 'audio' or 'audio_visual'
    'net_type': 'SELDnet',

    # data params
    'root_dir': './DCASE2025_2',  # parent directory containing the audio, video and labels directory
    'feat_dir': './DCASE2025_2/mel64_dnorm',  # store extracted features here

    'log_dir': 'logs',  # save all logs here like loss and metrics
    'checkpoints_dir': 'checkpoints',  # save trained model checkpoints and config
    'output_dir': 'outputs',  # save the predicted files here.

    # audio feature extraction params
    'sampling_rate': 24000,
    'hop_length_s': 0.0125,
    'nb_mels': 64,
    'max_freq': 2000,

    # video feature extraction params
    'fps': 10,
    'resnet_feature_size': 49,  # (7,7) feature_map for every frame from resnet

    # model params
    'model': 'seldnet',  # Model to use
    'nb_conv_blocks': 3,
    'nb_conv_filters': 64,
    'f_pool_size': [4, 4, 2],
    't_pool_size': [2, 2, 2],
    'dropout': 0.05,

    'rnn_size': 128,
    'nb_rnn_layers': 2,

    'nb_self_attn_layers': 2,
    'nb_attn_heads': 8,

    'nb_fnn_layers': 1,
    'fnn_size':128,

    'max_polyphony': 3,   # tracks for multiaccdoa
    'nb_classes': 13,
    'label_sequence_length': 50,  # 5 seconds with 100ms frames

    # loss params
    'multiACCDOA': True,
    'thresh_unify': 30,

    # training params
    'nb_epochs': 100,
    'batch_size': 64,
    'accum_batch': 64,
    'val_batch_size': 64,
    'nb_workers': 4,
    'shuffle': True,
    'persistent_workers': False,

    # optimizer params
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'scheduler': 'one',  # Learning rate scheduler type, one - OneCycleLR
    # folds for training, testing
    'dev_train_folds': ['fold1','fold2','fold3', 'fold5', 'fold6'],  # 'fold1,2' is the synthetic training data of DCASE2024. You can skip that if you do not use the synthetic data to train.
    # 'dev_train_folds': ['fold3'], # for finetune
    # 'dev_train_folds': ['fold3', 'fold5', 'fold6'], # fold5 for spatial scaper synth
    'dev_test_folds': ['fold4'], # for test

    # validation params
    'val_freq': 2,  # Validation frequency (every N epochs)
    'full': False,  # If True, validate each epoch

    # metric params
    'average': 'macro',                  # Supports 'micro': sample-wise average and 'macro': class-wise average.
    'segment_based_metrics': False,      # If True, uses segment-based metrics, else uses event-based metrics.
    'sed_thresh': 0.5,
    'lad_doa_thresh': 20,                # DOA error threshold for computing the detection metrics.
    'lad_dist_thresh': float('inf'),     # Absolute distance error threshold for computing the detection metrics.
    'lad_reldist_thresh': float('1.0'),  # Relative distance error threshold for computing the detection metrics.
    'lad_req_onscreen': False,           # Require correct on-screen estimation when computing the detection metrics.

    'use_jackknife': False,               # If True, uses jackknife to calc results of the best model on test/eval set.
                                          # CAUTION: Too slow to use jackknife

    # Distance Normalization Variables
    'dnorm': True,

    # If True, fine-tune the model on fold3 only (for dev_test)
    'finetune': False,  
    'val_batch_size':64,
}