OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6 python train.py --model_name stereo_only \
  --frame_ids 0 --use_stereo --png --width 640 --height 192 --batch_size 4 --model stereo --debug # --num_workers 0 #--load_weights_folder ./tmp/stereo_model_ori_v2/models/weights_0/ #--num_workers 0 #--scales 0, 1, 2
#   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7  python evaluate_depth.py --load_weights_folder ./tmp/stereo_right_v5/models/weights_5/ --eval_stereo --width 512 --height 256 --batch_size 8
#   cv2.imwrite("right_image.png",right_image_low[0].cpu().detach().numpy().transpose((1,2,0))*255)
#   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7  python evaluate_depth.py --load_weights_folder ./tmp/stereo_super/models/weights_8/ --eval_stereo --width 640 --height 192
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1  python evaluate_depth.py --load_weights_folder ./tmp/stereo_model_ori_v2/models/weights_0/ --eval_stereo --width 1280 --height 384
# s
# &   0.112  &   0.927  &   5.013  &   0.210  &   0.864  &   0.948  &   0.974  \\
# distill no crop
# &   0.119  &   0.852  &   4.681  &   0.204  &   0.858  &   0.950  &   0.977  \\
# distill crop
# 0.118  &   0.845  &   4.637  &   0.203  &   0.862  &   0.950  &   0.977  \\
# crop SSIM
# &   0.119  &   0.852  &   4.681  &   0.204  &   0.858  &   0.950  &   0.977  \\
# SSIM crop
# &   0.120  &   0.866  &   4.671  &   0.207  &   0.862  &   0.949  &   0.975  \\
# &   0.116  &   0.866  &   4.560  &   0.194  &   0.867  &   0.955  &   0.980  \\
# &   0.116  &   0.840  &   4.493  &   0.194  &   0.869  &   0.955  &   0.979  \\