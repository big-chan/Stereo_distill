OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python train.py --model_name stereo_only_v2 \
  --frame_ids 0 --use_stereo --png --width 640 --height 192 --batch_size 4 --model stereo --debug # --num_workers 0 #--load_weights_folder ./tmp/stereo_model_ori_v2/models/weights_0/ #--num_workers 0 #--scales 0, 1, 2
#   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7  python evaluate_depth.py --load_weights_folder ./tmp/stereo_right_v5/models/weights_5/ --eval_stereo --width 512 --height 256 --batch_size 8
