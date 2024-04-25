echo "start"
python3 train_model.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/DukeMTMC-reID/logs/baseline_scratch --dataset dukemtmc --model baseline --log_wandb 1 --run_name duke_baseline
python3 train_model.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/DukeMTMC-reID/logs/siamese_scratch --dataset dukemtmc --model siamese --log_wandb 1 --run_name duke_siamese
python3 train_model.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/DukeMTMC-reID/logs/osnet_scratch --dataset dukemtmc --model osnet_x0_25 --log_wandb 1 --run_name duke_osnet
python3 train_model.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/market1501/logs/osnet_scratch --dataset market1501 --model osnet_x0_25 --log_wandb 1 --run_name market_osnet
cp -R /home/ronak/datasets/DukeMTMC-reID/logs/baseline_scratch /home/ronak/datasets/market1501/logs/baseline_transfer_learning
cp -R /home/ronak/datasets/DukeMTMC-reID/logs/siamese_scratch /home/ronak/datasets/market1501/logs/siamese_transfer_learning
cp -R /home/ronak/datasets/DukeMTMC-reID/logs/osnet_scratch /home/ronak/datasets/market1501/logs/osnet_transfer_learning
cp -R /home/ronak/market1501/logs/osnet_scratch /home/ronak/datasets/DukeMTMC-reID/logs/osnet_transfer_learning
python3 finetune.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/DukeMTMC-reID/logs/baseline_transfer_learning --dataset dukemtmc --model baseline --log_wandb 1 --run_name duke_baseline_transfer_learning --max_epochs 10
python3 finetune.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/market1501/logs/baseline_transfer_learning --dataset market1501 --model baseline --log_wandb 1 --run_name market_baseline_transfer_learning --max_epochs 10

python3 finetune.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/DukeMTMC-reID/logs/siamese_transfer_learning --dataset dukemtmc --model siamese --log_wandb 1 --run_name duke_siamese_transfer_learning --max_epochs 10
python3 finetune.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/market1501/logs/siamese_transfer_learning --dataset market1501 --model siamese --log_wandb 1 --run_name market_siamese_transfer_learning --max_epochs 10

python3 finetune.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/DukeMTMC-reID/logs/osnet_transfer_learning --dataset dukemtmc --model osnet_x0_25 --log_wandb 1 --run_name duke_osnet_transfer_learning --max_epochs 10
python3 finetune.py --data_dir /home/ronak/datasets/ --logs_dir /home/ronak/datasets/market1501/logs/osnet_transfer_learning --dataset market1501 --model osnet_x0_25 --log_wandb 1 --run_name market_osnet_transfer_learning --max_epochs 10

echo "done"