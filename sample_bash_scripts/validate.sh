python3 validate.py          \
  --model BridgeNet          \
  --weights_loc results/SIG17_5L5S64U/stage2/53/SIG17_5L5S64U.tf \
  --model_name SIG17_5L5S64U \
  --dataset SIG17            \
  --image_type flow_corrected \
  --gpu_num -1                \
  --val_downsample 1 | tee SIG17_5L5S6U_SIG17val.txt