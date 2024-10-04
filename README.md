# RobIR: Robust Inverse Rendering for High-Illumination Scenes

## [Project page](https://ingra14m.github.io/RobIR_website) | [Paper](https://arxiv.org/abs/2310.13030) | [Data](https://drive.google.com/drive/folders/1maQVCc7xTxv9NYmWxLFT3bu0M9J4XhK0?usp=sharing)



## News

- **[10/03/2024]** Project page has been released.
- **[9/26/2024]** RobIR (formerly known as SIRe-IR) has been accepted by NeurIPS 2024. We will release the code these days.



## Dataset

In our paper, we use:

- synthetic dataset from [NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) and our [RobIR dataset](https://drive.google.com/drive/folders/1maQVCc7xTxv9NYmWxLFT3bu0M9J4XhK0?usp=sharing).
- real-world dataset from [NeuS](https://www.dropbox.com/sh/w0y8bbdmxzik3uk/AAAaZffBiJevxQzRskoOYcyja?dl=0).

We organize the datasets as follows:

```
├── data
│   | nerf 
│     ├── hotdog
│     ├── lego 
│     ├── ...
│   | robir_dataset
│     ├── truck
│     ├── chessboard
│     ├── ...
│   | neus
│     ├── bear
│     ├── sculpture
│     ├── ...
```

## Run

### Environment

- Set up the Python environment

```shell
conda create -n sire-env python=3.7
conda activate robust-ir-env

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install -r requirement.txt
```



### Train

#### Stage 1: NeuS (Geometry Prior)

```shell
python neus/exp_runner.py --gin_file neus/config/neus-hash.gin
```



#### Stage 2: BRDF Estimation

**2.1 Train Norm**

```shell
python training/exp_runner.py 
          --conf confs_sg/hotdog.conf
          --data_split_dir data/hotdog
          --expname hotdog
          --trainstage Norm
```

**2.2 Train Visibility and Indirect Illumination**

```shell
python training/exp_runner.py 
          --conf confs_sg/hotdog.conf
          --data_split_dir data/hotdog
          --expname hotdog
          --trainstage Vis
```

**2.3 Train PBR**

```shell
python training/exp_runner.py 
          --conf confs_sg/hotdog.conf
          --data_split_dir data/hotdog
          --expname hotdog
          --trainstage PBR
```

**2.4 Train Reg-Estim**

```shell
python training/exp_runner.py 
          --conf confs_sg/hotdog.conf
          --data_split_dir data/hotdog
          --expname hotdog
          --trainstage CESR
```



## Results

### Albedo

<img src="assets/albedo.png" alt="image-20231020012659356" style="zoom:50%;" />

### Roughness

<img src="assets/roughness.png" alt="image-20231020012659356" style="zoom:50%;" />

### Envmap

<img src="assets/envmap.png" alt="image-20231020012659356" style="zoom:50%;" />

### Relighting

<img src="assets/relighting.png" alt="image-20231020012659356" style="zoom:50%;" />

### De-shadow

<img src="assets/deshadow.png" alt="image-20231020012659356" style="zoom:50%;" />

## BibTex

```
@article{yang2023sireir,
    title={SIRe-IR: Inverse Rendering for BRDF Reconstruction with Shadow and Illumination Removal in High-Illuminance Scenes},
    author={Yang, Ziyi and Chen, Yanzhen and Gao, Xinyu and Yuan, Yazhen and Wu, Yu and Zhou, Xiaowei and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2310.13030},
    year={2023}
}
```
