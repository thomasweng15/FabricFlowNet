# FabricFlowNet: Bimanual Cloth Manipulation with a Flow-based Policy
Thomas Weng, Sujay Bajracharya, Yufei Wang, Khush Agrawal, David Held
Conference on Robot Learning 2021

[[arXiv](https://arxiv.org/abs/2111.05623)] [[pdf](https://arxiv.org/pdf/2111.05623.pdf)] [[project page](https://sites.google.com/view/fabricflownet)]

This repo contains the code for running the FabricFlowNet model used to report the simulation results in the paper.

The ROS code for running FabricFlowNet on a real-world system will be released in a separate repository. 

## File Structure
```angular2html
├── FabricFlowNet/
|   |── data/                   # Folder for datasets, saved runs, evaluation goals 
|   └── fabricflownet/
|       |── flownet/            # FlowNet model and training code
|       |── picknet/            # PickNet model and training code
|       |── eval.py             # Evaluation script
|       └── utils.py            
└── softgym/                    # SoftGym submodule
```

## Installation

These instructions have been tested on Ubuntu 18.04 with NVIDIA GTX 3080/3090 GPUs. 
SoftGym requires CUDA 9.2+, FFN training and inference have been tested on CUDA 11.1. 

* Clone this repo
* Set up the [softgym](https://github.com/Xingyu-Lin/softgym) submodule, tracking the `fabricflownet` branch: `git submodule update --init`
* Compile PyFlex: `. prepare_1.0.sh` and `. compile_1.0.sh`. Check the compile script to make sure that the `CUDA_BIN_PATH` env variable is set to the path of the CUDA library you installed SoftGym with. More detailed Softgym installation instructions can be found in the original [repo](https://github.com/Xingyu-Lin/softgym).
* In the FabricFlowNet directory, activate the conda environment and set environment variables: `. prepare_1.0.sh` 
* `conda env update -f environment.yml --prune`. Ensure that you install a PyTorch version that is suitable for your CUDA version.
* Install the repo as a python package: `pip install -e .`

# Evaluation
* Download the evaluation set and model weights into `./data/`:
    * Download and extract the [evaluation set](https://drive.google.com/file/d/1A9GUPXuVIC1K-LCCvzrK95-m9_UVSbPd/view?usp=sharing)
    * Download the [FlowNet weights](https://drive.google.com/file/d/1P7Upskczb-iqOsPjgcjsd4cnQQEuf-uY/view?usp=sharing), this does not need to be extracted
    * Download and extract the [PickNet weights](https://drive.google.com/file/d/1dCuSpMyvzkPU3AL7MeXeL7knP5ngyKvq/view?usp=sharing)
* Run the evaluation script: `python fabricflownet/eval.py --run_name=data/picknet_run --checkpoint=105000 --goals=square_towel`
    * To run in headless mode, add the `--headless` flag; use the `-h` flag to see other available flags. 
* Performance on square towel goals from the paper (in mm):
```
Square Towel (mm)
all: 6.803 +/- 10.413
one-step: 4.262 +/- 2.287
mul-step: 23.741 +/- 21.597
```
```
Rectangular Towel (mm)
all: 9.270 +/- 7.001
one-step: 4.254 +/- 1.086
mul-step: 16.793 +/- 5.143
```
Performance on t-shirt goals will be available soon. 


## Citation
If you find this code useful in your research, please feel free to cite:
```
@inproceedings{weng2021fabricflownet,
 title={FabricFlowNet: Bimanual Cloth Manipulation with a Flow-based Policy},
 author={Weng, Thomas and Bajracharya, Sujay and Wang, Yufei and Agrawal, Khush and Held, David},
 booktitle={Conference on Robot Learning},
 year={2021}
}
```
