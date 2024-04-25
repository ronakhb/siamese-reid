# encoding: utf-8

import argparse
import os
import sys
import torch
from torch.backends import cudnn
import numpy as np

# Add current directory to the Python path
sys.path.append(".")

# Import functions and classes from the project modules
from data import make_data_loader_market as make_data_loader
from engine.trainer import do_train_market as do_train
from modeling import build_model
from loss import make_loss_with_center
from solver import make_optimizer, WarmupMultiStepLR
from engine.inference import inference
import datetime
import onnxruntime as ort

# Function to count the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to load pretrained weights into the model if available
def load_network_pretrain(model, cfg):
    # Check if the checkpoint file exists
    path = os.path.join(cfg.logs_dir, "checkpoint.pth")
    if not os.path.exists(path):
        return model, 0, 0.0
    # Load pretrained weights from the checkpoint file
    pre_dict = torch.load(path)
    model.load_state_dict(pre_dict["state_dict"],strict=False)
    start_epoch = pre_dict["epoch"]
    best_acc = pre_dict["best_acc"]
    print("start_epoch:", start_epoch)
    print("best_acc:", best_acc)
    return model, start_epoch, best_acc

# Main function
def main(cfg):
    # Make data loaders for the specified dataset
    dataset, train_loader, test_loader, num_query, num_classes = make_data_loader(cfg)
    num_classes = 751  # Override num_classes if necessary
    
    # Build the model
    model = build_model(
        num_classes, cfg.model_name, pretrain_choice=True
    )  # num_classes=5000
    model = torch.nn.DataParallel(model).cuda() if torch.cuda.is_available() else model
    
    # Print the number of parameters in the model
    if cfg.param_count == 1:
        num_params = count_parameters(model)
        num_params /= 1000000
        print("Model has ",num_params," million params")
        exit(0)

    # Define loss function based on the model architecture
    if cfg.model_name == "siamese":
        loss_func,_ = make_loss_with_center(cfg, num_classes,feat_dim=960)  # MobilenetaLarge Model
    elif cfg.model_name == "baseline":
        loss_func,_ = make_loss_with_center(cfg, num_classes,feat_dim=2048)  # Resnet Baseline
    elif cfg.model_name.startswith("osnet"):
        loss_func,_ = make_loss_with_center(cfg, num_classes,feat_dim=1000) #OSNet
    else:
        loss_func,_ = make_loss_with_center(cfg, num_classes,feat_dim=576)  # MobilenetSmall Model
    
    # Make optimizer and scheduler
    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.steps,
        cfg.gamma,
        cfg.warmup_factor,
        cfg.warmup_iters,
        cfg.warmup_method,
    )

    if cfg.train == 1:
        start_epoch = 0
        acc_best = 0.0
        if cfg.resume == 1:
            model, start_epoch, acc_best = load_network_pretrain(model, cfg)

        # Train the model
        do_train(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_func, num_query, start_epoch, acc_best)
        
        # Load the best model and perform inference
        last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
        model.load_state_dict(last_model_wts['state_dict'],strict=False)
        mAP, cmc1, cmc5, cmc10, cmc20, feat_dict = inference(model, test_loader, num_query, True,save_cmc_plot=True,plot_name="/home/ronak/PRCV final project/siamese_reid/plots/" + cfg.run_name)

        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        print('{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP))
    else:
        # Test
        last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint.pth'))
        model.load_state_dict(last_model_wts['state_dict'],strict=False)
        mAP, cmc1, cmc5, cmc10, cmc20, feat_dict = inference(model, test_loader, num_query, True,use_cosine=cfg.use_cosine,save_cmc_plot=False, plot_name="/home/ronak/PRCV final project/siamese_reid/plots/" + cfg.run_name)

        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        print('{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP))
    
    return

if __name__ == "__main__":
    # Set CUDA visible devices
    gpu_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cudnn.benchmark = True

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    # Add arguments for various parameters
    
    # DATA
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_per_id", type=int, default=4)
    parser.add_argument("--batch_size_test", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height_mask", type=int, default=256)
    parser.add_argument("--width_mask", type=int, default=128)

    # MODEL
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--model_name", type=str, default="siamese")

    # OPTIMIZER
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--lr_center", type=float, default=0.5)
    parser.add_argument("--center_loss_weight", type=float, default=0.0005)
    parser.add_argument("--steps", type=list, default=[40, 70])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--cluster_margin", type=float, default=0.3)
    parser.add_argument("--bias_lr_factor", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--weight_decay_bias", type=float, default=5e-4)
    parser.add_argument("--range_k", type=float, default=2)
    parser.add_argument("--range_margin", type=float, default=0.3)
    parser.add_argument("--range_alpha", type=float, default=0)
    parser.add_argument("--range_beta", type=float, default=1)
    parser.add_argument("--range_loss_weight", type=float, default=1)
    parser.add_argument("--warmup_factor", type=float, default=0.01)
    parser.add_argument("--warmup_iters", type=float, default=10)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--optimizer_name", type=str, default="SGD", help="Adam, SGD")
    parser.add_argument("--momentum", type=float, default=0.95)

    # TRAINER
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--train", type=int, default=1)  # change train or test mode
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--num_works", type=int, default=16)
    parser.add_argument("--log_wandb", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="")

    # misc
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--param_count", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="market1501")
    parser.add_argument("--check_onnx", type=int, default=0)
    parser.add_argument("--use_cosine", action='store_true')
    parser.add_argument(
        "--data_dir", type=str, default="/home/ronak/datasets/"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="/home/ronak/datasets/market1501/logs/baseline"
    )

    # Parse arguments
    cfg = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(cfg.logs_dir):
        os.makedirs(cfg.logs_dir)
    
    # Call the main function
    main(cfg)
