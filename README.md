# Project Description
Team Member(s) - Ronak Harish Bhanushali
This project tests out different models and datasets for person reidentification and evaluates its performance. Models used for this are - 
1. ResNet50 Baseline as shown in Bag of Tricks Paper
2. OSNet
3. Custom MobilenetV3 Large backbone model

All the training data for this can be found on wandb on this link - https://wandb.ai/attention-boys/Mobilenet%20ReID%20LaST%20Script/workspace?nw=nwuserbhanushaliron
This repository uses the training pipeline from https://github.com/shuxjweb/last
Functions in the code are modified to enable wandb logging as well as new plotting functions and distance metrics are used. Additionally we have the train_model.py script, finetune.py, autorunner.sh and auto_evaluator.sh files added.  

## Environment Setup

Create a new pytorch environment using the given siamese_net.yml file. Run the following command
```bash
conda env create -f environment.yml
conda activate pytorch
```

## Usage

To train the person ReID model, run the `train_model.py` script with appropriate command-line arguments specifying the training configuration. For example:

## Training

```bash
python train_model.py --batch_size 64 --lr 0.00035 --model_name siamese --max_epochs 50 --train 1 --logs_dir /home/ronak/data/logs --dataset market1501 --log_wandb 1 --run_name siamese_market --data_dir /home/ronak/data/
```
## Testing

```bash
python3 train_model.py --model baseline --train 0 --dataset market1501  --logs_dir /home/ronak/datasets/market1501/logs/baseline --data_dir /home/ronak/datasets/
```
## Arguments passed

### logs_dir
Path to store checkpoints.

### run_name
Identifier for the run in Weights & Biases and also used for saving plots.

### model
Select from "siamese" for a custom MobileNet-based network, "baseline" for ResNet50 backbone, and "osnet_x0_25" for OSNet with a 0.25 multiplier. Check the available models for other options.

### dataset
Choose from "market1501", "dukemtmc", and "last". The code supports additional datasets not explored in this project; refer to the datasets `__init__` file for more information on supported datasets.

### data_dir
Path to the selected dataset.


## Presentation Link - https://drive.google.com/drive/folders/1RfEx5Dv7uTjemIni9CiiSmViQzFH0fJi?usp=sharing