OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6 python train.py --model_name stereo_pwc_high_3m --eval_split kitti_2015 --split kitti_2015 --eval_stereo \
  --frame_ids 0 --use_stereo --png --width 1024 --height 320 --batch_size 8 --model stereo --debug --using_detach --num_epochs 50 
  
  
  #--load_weights_folder ./tmp/stereo_only_v2/models/weights_2/ # --num_workers 0 #--load_weights_folder ./tmp/stereo_model_ori_v2/models/weights_0/ #--num_workers 0 #--scales 0, 1, 2
#   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1  python evaluate_depth.py --load_weights_folder ./tmp/stereo_only_pwc6_mono1/models/weights_19/ --eval_stereo --width 640 --height 192 --batch_size 8 --model stereo
