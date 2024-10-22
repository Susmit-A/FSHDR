# Few-shot Deep HDR Deghosting
This Repository contains code and pretrained models for our paper: [Labeled from Unlabeled: Exploiting Unlabeled Data for Few-shot Deep HDR Deghosting](https://openaccess.thecvf.com/content/CVPR2021/papers/Prabhakar_Labeled_From_Unlabeled_Exploiting_Unlabeled_Data_for_Few-Shot_Deep_HDR_CVPR_2021_paper.pdf) accepted at CVPR, 2021.

It has been tested on RTX 2080Ti with Tensorflow 2.5.0. It also requires MATLAB for performing optical flow correction. Some parts of the flow correction code have been taken from [here](http://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR).

![Banner](github_images/overview.png)


## Installation and Setup
### Docker Environment:
Getting base image:
```shell script
$ docker pull tensorflow/tensorflow:2.5.0-gpu
```

Running base image:
```shell script
$ docker run --rm -it tensorflow/tensorflow:2.5.0-gpu bash
```

Installing dependencies:
```shell script
(docker)# apt update
(docker)# apt install -y ffmpeg libsm6 libxext6 libxrender-dev
(docker)# pip install opencv-python
```

Using MATLAB:

The MATLAB commands provided in this document work outside the docker container. If you still want to create a docker image with MATLAB installed, please follow the instructions from https://github.com/mathworks-ref-arch/matlab-dockerfile.

### Dataset Preparation:

1. For Kalantari dataset (SIG17), run the following from the `dataset` folder:
```shell script
$ wget https://www.robots.ox.ac.uk/~szwu/storage/hdr/kalantari_dataset.zip
$ unzip kalantari_dataset.zip
$ python prepare_SIG17.py
$ rm -rf train test
$ matlab -nodisplay -nosplash -nodesktop -r "FlowCorrectAndStoreBackFlows('SIG17'); exit;"
```

2. For Prabhakar dataset (ICCP19), from the `dataset` folder,
##### Download ICCP19 Dataset from https://www.kaggle.com/valrnn/iccp19
Extract ```train``` and ```val``` to ```ICCP19``` folder and then run the following:

```shell script
$ matlab -nodisplay -nosplash -nodesktop -r "FlowCorrectAndStoreBackFlows('ICCP19'); exit;"
```

## Steps for running different components
1. Download atleast one of the datasets and prepare them.
2. Start a docker container.
3. Run the scripts given below inside the container.

### Training and Artificial Labeled Data Synthesis
1. Run ```sample_bash_scripts/train_S1.sh``` for Stage 1 training. Significant parameters are model, num_static, num_SCL_dynamic, num_supervised_dynamic, dataset, gpu_num, and model_name.

2. Run ```sample_bash_scripts/predict_synthetic_HDRs.sh``` for generating artificial labeled dataset from Stage 1 trained model. Ensure all parameters are proper and point to correct locations.

3. Run ```sample_bash_scripts/train_S2.sh``` for Stage 2 training. Significant parameters worth modifying are model, resume_weights_loc, num_synthetic_dynamic, dataset, gpu_num, and model_name.

### Inference
Run ```sample_bash_scripts/validate.sh``` for validating provided models. Ensure all parameters are proper and point to correct locations.

## Provided Model Weights
For each of zero-shot, one-shot, five-shot, and fully supervised runs with SIG17 dataset, we share BridgeNet model weights in ```results```.

## Citation
When citing this work, you should use the following Bibtex:

    @inproceedings{prabhakar2021labeled,
      title={Labeled from Unlabeled: Exploiting Unlabeled Data for Few-shot Deep HDR Deghosting},
      author={Prabhakar, K Ram and Senthil, Gowtham and Agrawal, Susmit and Babu, R Venkatesh and Gorthi, Rama Krishna Sai S},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={4875--4885},
      year={2021}
    }
