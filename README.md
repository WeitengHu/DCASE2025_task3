# Baseline Model for DCASE 2025 Task3

This repository implements the audio-only system of DCASE 2025 Task3. 
## Setup
This setup has been tested using Python 3.12.11 and Torch 2.8.0+cu126. We have trained this model on a 3090 GPU.

First, We recommend to create a new environment and install the necessary packages.
```
conda create --name dcase2025_task3 python=3.12.11
conda activate dcase2025_task3
pip install -r requirements.txt
```

## Quick Start
### 下载数据集
run
```
bash scripts/download.sh
```
to download DCASE2025 official dataset to `./DCASE2025` .

### 解压
run 
```
bash scripts/unzip.sh
```
to unzip the zip files and delete the original zip files

### 预处理（可选）
run
```
bash scripts/left_right_swap.sh
```
to swap the left and right channel for dataset.

You can also add your own synthetic datasets. The directory structure should be:

```bash
./DCASE2025/
├── stereo_dev/
│   ├── dev-train-tau/*.wav
│   ├── dev-train-sony/*.wav
│   ├── dev-train-realrc/*.wav (to be created by `left_right_swap.py`)
│   ├── dev-train-synth/*.wav (sythetic dataset)
│   ├── dev-test-tau/*.wav
│   ├── dev-test-sony/*.wav
├── metadata_dev/
│   ├── dev-train-tau/*.csv
│   ├── dev-train-sony/*.csv
│   ├── dev-train-realrc/*.csv (to be created by `left_right_swap.py`)
│   ├── dev-train-synth/*.csv (sythetic dataset)
│   ├── dev-test-tau/*.csv
│   ├── dev-test-sony/*.csv
```

### 训练
run
```
bash scripts/main.sh
```
You can modify parameters in `scripts/main.sh` and `src/parameters.py`.

To set up `main.sh`, for example:

```
python3 main.py --ms --iv --gamma --itfm --exp "experiment_name"
```
`--ms --iv --gamma` means to use Mid-Side (MS) spectrograms, Mid-Side Intensity Vector (IV), and the Magnitude-Squared Coherence (MSC) respectively. You can add or remove the features as necessary.

`--compfreq` means to use Frequence Shifting and FilterAugment. `--itfm` means to use ITFM (Inter-Channel-Aware Time-Frequency Masking). Please feel free to add/remove the arguments as necessary. 

`--exp` is for your name of the experiment.



### 微调(可选)
First, you should modify parameters in `scripts/finetune.sh`. Please ensure the input features are the same as `scripts/main.sh`.

`--pretrained_exp` is your model file name of your pretrained model saved in `./checkpoints`. Run
```
bash scripts/finetune.sh
```

### 比较结果
To compare the results of your experiments, directly run 
```bash scripts/compare.sh```.


## References and Acknowledgement
This repository is based on the [DCASE 2025 Yeow NTU Result](https://github.com/itsjunwei/NTU_SNTL_Task3)
```
@techreport{Yeow_NTU_task3a_report,
    Author = "Yeow, Jun-Wei and Tan, Ee-Leng and Peksi, Santi and Gan, Woon-Seng",
    title = "IMPROVING STEREO 3D SOUND EVENT LOCALIZATION AND DETECTION: PERCEPTUAL FEATURES, STEREO-SPECIFIC DATA AUGMENTATION, AND DISTANCE NORMALIZATION",
    institution = "DCASE2025 Challenge",
    year = "2025"
}
```