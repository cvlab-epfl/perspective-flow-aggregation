CUDA_VISIBLE_DEVICES=0 python3 train.py --name raft-occlinemod --stage occlinemod --validation occlinemod --gpus 0 --num_steps 100000 --batch_size 8 --lr 0.0001 --image_size 256 256 --wdecay 0.0001
