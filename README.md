# SIRe-IR: Inverse Rendering for BRDF Reconstruction with Shadow and Illumination Removal in High-Illuminance Scenes

## [Project page (Coming Soon...)](https://ingra14m.github.io/Deformable-3D-Gaussians.github.io/) | [Paper]()

<img src="assets/teaser.png" alt="image-20231020012408139" style="zoom:50%;" />

This repository contains the official implementation associated with the paper "SIRe-IR: Inverse Rendering for BRDF Reconstruction with Shadow and Illumination Removal in High-Illuminance Scenes". We will release our code after we sort it out.



## Preparation

- Set up the Python environment

```shell
conda create -n hdrfactor-env python=3.7
conda activate hdrfactor-env

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install -r requirement.txt
```





## Results

<img src="assets/results.png" alt="image-20231020012659356" style="zoom:50%;" />

