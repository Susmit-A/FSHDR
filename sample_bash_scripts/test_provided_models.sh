python3 validate.py          \
  --weights_loc results/zeroshot_SIG17/UCSD_Ours_FS0_S3R3.tf \
  --model_name zeroshot_SIG17 \
  --dataset SIG17             \
  --rtx_mixed_precision       \
  --gpu_num 0 | tee zeroshot_SIG17.txt
python3 validate.py          \
  --weights_loc results/oneshot_SIG17/UCSD_Ours_FS1_S3R2.tf \
  --model_name oneshot_SIG17 \
  --dataset SIG17            \
  --rtx_mixed_precision       \
  --gpu_num 0 | tee oneshot_SIG17.txt
python3 validate.py          \
  --weights_loc results/fiveshot_SIG17/UCSD_Ours_FS5_S3R4.tf \
  --model_name fiveshot_SIG17 \
  --dataset SIG17            \
  --rtx_mixed_precision       \
  --gpu_num 0 | tee fiveshot_SIG17.txt