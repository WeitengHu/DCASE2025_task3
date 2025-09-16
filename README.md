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

## Dataset
Official Dataset of DCASE 2025 Task 3 can be found at [here](https://zenodo.org/records/15559774).

Optionally, you can use additional synthetic dataset.  You can utilize the publicly released [DCASE 2024 simulated dataset](https://zenodo.org/records/10932241) or use [SpatialScaper](https://github.com/iranroman/SpatialScaper) to generate addtional FOA format data. Then, you can transfer the FOA audio to stereo audio through [stereo SELD data generator](https://github.com/SonyResearch/dcase2025_stereo_seld_data_generator)

Additionally, we applied channel swapping onto the real recordings. You can use `left_right_swap.py` for this purpose.

The directory structure should be:

```bash
DCASE2025_SELD_dataset/
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

## Feature
Within this work, we use a set of perceptually-motivated input features, including Mid-Side (MS) spectrograms, Mid-Side Intensity Vector (IV), and the Magnitude-Squared Coherence (MSC) between the stereo channels. 

## Data Augmentation
First, we propose ACS method to swap the left and right channels. You can use `left_right_swap.py` for this purpose.

We provide other data augmentation techniques in `training_utils.py`, including ITFM, Frequence Shifting and FilterAugment.

## Getting Start
All parameters are stored in `parameters.py`, and you can change them for your own experiments.

To train a model yourself, setup  `parameters.py` and `main.sh`, and then directly run

```bash
bash main.sh
```

To set up `main.sh`, for example:

```
python3 main.py --ms --iv --gamma --compfreq --itfm --exp "experiment_name"
```
`--ms --iv --gamma` means to use Mid-Side (MS) spectrograms, Mid-Side Intensity Vector (IV), and the Magnitude-Squared Coherence (MSC) respectively.

`--compfreq` means to use Frequence Shifting and FilterAugment. `--itfm` means to use ITFM (Inter-Channel-Aware Time-Frequency Masking).

`--exp` is for your name of the experiment.

Please feel free to add/remove the arguments as necessary. 


## References and Acknowledgement
This repository is based on the [DCASE 2025 NTU Result](https://github.com/itsjunwei/NTU_SNTL_Task3)
```
@techreport{Yeow_NTU_task3a_report,
    Author = "Yeow, Jun-Wei and Tan, Ee-Leng and Peksi, Santi and Gan, Woon-Seng",
    title = "IMPROVING STEREO 3D SOUND EVENT LOCALIZATION AND DETECTION: PERCEPTUAL FEATURES, STEREO-SPECIFIC DATA AUGMENTATION, AND DISTANCE NORMALIZATION",
    institution = "DCASE2025 Challenge",
    year = "2025"
}
```