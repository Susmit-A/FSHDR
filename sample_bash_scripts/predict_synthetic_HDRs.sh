python3 predict_synthetic_HDRs.py                                   \
  --model BridgeNet                                                 \
  --weights_loc results/SIG17_5L5S64U/stage1/59/SIG17_5L5S64U.tf    \
  --model_name SIG17_5L5S64U                                        \
  --dataset SIG17                                                   \
  --image_type flow_corrected                                       \
  --rtx_mixed_precision                                             \
  --gpu_num 0 | tee SIG17_5L5S6U_SIG17synth.txt
  
matlab -nodisplay -nosplash -nodesktop -r \
    "addpath('artificial_synthesis'); ApplyReverseFlow('SIG17', 'SIG17_5L5S64U', 'dataset/temp_datasets/SIG17_5L5S64U'); exit;"