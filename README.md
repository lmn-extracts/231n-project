# 231n-project
Capstone Project for CS 231N: Convolutional Neural Networks for Visual Recognition

Typical command to run training

python ./train.py --exp_name mjsynthMay28TrainEval --lr 0.1 --lr_decay_steps 1000  --lr_decay_rate 0.99  --datadir /apps/afassa/data/mjsynth-tfrec/ --val_batch_size 3584 --batch_size 256 --eval_steps 5 --train_steps 74000 --eval_throttle_secs 600 --parallel_cpu 32


