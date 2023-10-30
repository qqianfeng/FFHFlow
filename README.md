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

The last step is to install prohmr as a Python package. This will allow you to import it from anywhere in your system.
Since you might want to modify the code, we recommend installing as follows:
```
python setup.py develop
```

## Adaptation to normflows package

### Modifies in normflow package

1. [?]: Conditional Glow: Adapt input from images to 1-dim bps input inside Glow init function

2. [x]: Conditional Glow: Replace Conv2d with ResidualNet

3. []: Modify AffineCouplingBlock with context

4. []: Do we need to modify Invertible1x1Conv?

5. [x?]: Modify ActNorm

6. [x?]: Modify GMM base distribution

7. [?]: modify ConditionalNormalizingFlow class

### Modifies in ffhflow

1. []: change apis to conditional glow, forward/inverse, loss etc.
