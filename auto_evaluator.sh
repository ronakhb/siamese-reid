echo "siamese"
echo "Last"
python3 train_model.py --model siamese --train 0 --dataset last  --logs_dir /home/ronak/datasets/market1501/logs/siamese --data_dir /home/ronak/datasets/last/
echo "Duke"
python3 train_model.py --model siamese --train 0 --dataset dukemtmc  --logs_dir /home/ronak/datasets/DukeMTMC-reID/logs/siamese_scratch --data_dir /home/ronak/datasets/
echo "market"
python3 train_model.py --model siamese --train 0 --dataset market1501  --logs_dir /home/ronak/datasets/market1501/logs/siamese --data_dir /home/ronak/datasets/

echo "baseline"
echo "Last"
python3 train_model.py --model baseline --train 0 --dataset last  --logs_dir /home/ronak/datasets/market1501/logs/baseline --data_dir /home/ronak/datasets/last/
echo "Duke"
python3 train_model.py --model baseline --train 0 --dataset dukemtmc  --logs_dir /home/ronak/datasets/DukeMTMC-reID/logs/baseline_scratch --data_dir /home/ronak/datasets/
echo "market"
python3 train_model.py --model baseline --train 0 --dataset market1501  --logs_dir /home/ronak/datasets/market1501/logs/baseline --data_dir /home/ronak/datasets/