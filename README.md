# FFHFlow

## Installation

```
conda env create -f environment.yml
```

After the installation is complete you can activate the conda environment by running:
```
conda activate prohmr
```
Install missing packages from pip.
```
pip install -r requirements.txt
```
Installing customized nflow package:
``` 
git submodule update --init --recursive
```

``` 
cd dependencies/nflows & pip install -e .
```

The last step is to install prohmr as a Python package. This will allow you to import it from anywhere in your system.
Since you might want to modify the code, we recommend installing as follows:
```
python setup.py develop
```

if normflows packages cannot be properly installed
```
git clone https://github.com/qqianfeng/normalizing-flows.git
cd normalizing-flows
python setup.py install
```

## Download dataset
Check repo of [FFHNet](https://github.com/qianbot/FFHNet)

## Train

### Train ffhflow-cnf
```
python3 train.py --model_cfg ffhflow/configs/ffhflow_cnf.yaml 
```
### Train ffhflow-lvm
```
python3 train.py --model_cfg ffhflow/configs/ffhflow_lvm.yaml
```

## Eval

### Eval ffhflow-cnf
```
python3 eval.py --model_cfg checkpoints/flow_cnf_lr1e-4_RND2/hparams.yaml
```

### Eval lvffhflow-lvm
```
python3 eval.py --model_cfg checkpoints/flow_lvm_lr1e-4_RND-1_lx0260/hparams.yaml
```

## Folder structure
```
|── environment.yml
├── eval_batch.pth
├── eval_ffhflow_2models.py
├── eval_ffhflow_normflows.py
├── eval_ffhflow.py
├── ffhflow
│   ├── backbones
│   │   ├── ffhgenerator.py
│   │   ├── pointnet.py
│   │   └── vae.py
│   ├── configs
│   │   ├── local_inn.yaml
│   │   ├── prohmr.yaml
│   ├── datasets
│   │   ├── ffhgenerator_data_set.py
│   ├── ffhflow_normal_pos_enc.py
│   ├── ffhflow_normal.py
│   ├── ffhflow_pos_enc_neg_grasp.py     # flow with positional encoding and input of positive + negative grasps
│   ├── ffhflow_pos_enc.py               # flow with positional encoding for only rotation part
│   ├── ffhflow_pos_enc_with_transl.py   # flow with positional encoding for rotation + translation
│   ├── ffhflow.py
│   ├── heads
│   │   ├── local_inn.py
│   │   ├── normflows_rot_glow_pos_enc_with_transl.py # with normflow pacakge, flow with positional encoding for rotation + translation
│   │   ├── normflows_rot_glow.py                     # with normflow pacakge, conditional glow class
│   │   ├── rot_flow_normal_pos_enc.py
│   │   ├── rot_flow_normal.py
│   │   ├── rot_flow_pos_enc_neg_grasp.py       # flow with positional encoding and input of positive + negative grasps
│   │   ├── rot_flow_pos_enc.py                 # flow with positional encoding for only rotation part
│   │   ├── rot_flow_pos_enc_with_transl.py     # flow with positional encoding for rotation + translation
│   │   └── rot_flow.py
│   ├── normflows_ffhflow_pos_enc_with_transl.py    # with normflow pacakge, flow with positional encoding for rotation + translation
│   └── utils
│       ├── definitions.py
│       ├── grasp_data_handler.py
│       ├── metrics.py
│       ├── train_utils.py
│       ├── utils.py
│       ├── vis_angle_vector.py
│       └── visualization.py
├── README.md
├── requirements.txt
├── setup.py
├── train_ffhflow_normflows.py
└── train_ffhflow.py
```