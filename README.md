# Stereo_distill

## Setup

```
ln -s /data/datasets/KITTI_raw ./kitti_data
```

## train

```
bash train.sh
```
## eval

```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0  python evaluate_depth.py --load_weights_folder ./tmp/stereo_only/models/weights_0/ --eval_stereo --width 512 --height 256 --batch_size 4 --model stereo

```
