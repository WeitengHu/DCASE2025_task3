"""
distance_normalization.py

This module provides functionality to automatically calculate distance normalization parameters
from the training dataset.

"""

import glob
import os
import pandas as pd
import numpy as np
from rich.progress import Progress


def calculate_distance_normalization_params(root_dir):
    """
    Calculate distance normalization parameters (d_mean, d_std, d_max) from training data.
    
    Args:
        root_dir (str): Root directory containing metadata_dev folder
        train_folds (list): List of training folds to include. If None, uses all available folds.
    
    Returns:
        dict: Dictionary containing d_mean, d_std, and d_max values
    """
    
    # Define default training folds if none provided

        # Get all available folds
        # sony_files = glob.glob(os.path.join(root_dir, 'metadata_dev', 'dev-train-sony', '*.csv'))
        # tau_files = glob.glob(os.path.join(root_dir, 'metadata_dev', 'dev-train-tau', '*.csv'))
        # realcs_files = glob.glob(os.path.join(root_dir, 'metadata_dev', 'dev-train-realcs', '*.csv'))
        # ss_files = glob.glob(os.path.join(root_dir, 'metadata_dev', 'dev-train-ss', '*.csv'))
        
        # label_files = sony_files + tau_files + realcs_files + ss_files
    label_files = glob.glob(os.path.join(root_dir, 'metadata_dev', 'dev-*', '*.csv'))

    # Remove duplicates
    label_files = list(set(label_files))
    
    if not label_files:
        raise ValueError(f"No label files found in {root_dir}/metadata_dev for the specified folds")
    
    print(f"Found {len(label_files)} label files for distance normalization calculation")
    
    # Extract all distance values
    all_distances = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing label files for distance normalization...", total=len(label_files))
        for file in label_files:
            try:
                df = pd.read_csv(file)
                if 'distance' in df.columns:
                    raw_distance = (df['distance'] / 100.0).tolist()  # Convert from cm to m
                    all_distances.extend(raw_distance)
            except Exception as e:
                print(f"Warning: Could not process file {file}: {e}")
            progress.update(task, advance=1)
    
    if not all_distances:
        raise ValueError("No distance values found in the label files")
    
    print(f"Extracted {len(all_distances)} distance values")
    
    # Calculate normalization parameters
    all_distances = np.array(all_distances)
    
    # Calculate mean and std
    mean_dist = np.mean(all_distances)
    std_dist = np.std(all_distances)
    
    # Standardize distances
    d_stand = (all_distances - mean_dist) / std_dist
    
    # Calculate d_max
    d_max = np.max(np.abs(d_stand))  # Use absolute max for symmetric normalization
    
    normalization_params = {
        'd_mean': float(mean_dist),
        'd_std': float(std_dist),
        'd_max': float(d_max)
    }
    
    print(f"Calculated distance normalization parameters:")
    print(f"  d_mean: {normalization_params['d_mean']:.6f}")
    print(f"  d_std: {normalization_params['d_std']:.6f}")
    print(f"  d_max: {normalization_params['d_max']:.6f}")
    
    return normalization_params

# calculate_distance_normalization_params("/mnt/hwt/DCASE2025")