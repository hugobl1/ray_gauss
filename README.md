<p align="center">

  <h1 align="center">RayGauss: Volumetric Gaussian-Based Ray Casting for Photorealistic Novel View Synthesis</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/hugo-blanc-a2b46016a/">Hugo Blanc</a>
    ·
    <a href="https://scholar.google.com/citations?user=zR1n_4QAAAAJ&hl=fr">Jean-Emmanuel Deschaud</a>
    ·
    <a href="https://scholar.google.fr/citations?user=3eO15d0AAAAJ&hl=fr">Alexis Paljic</a>

  </p>
  <h2 align="center">WACV 2025</h2>

  <h3 align="center"><a href="https://drive.google.com/file/d/1qbJjbScbUJOKoYc0iLhk1NE7rtcHp8lH/view?usp=sharing">Paper</a> | <a href="https://arxiv.org/pdf/2408.03356">arXiv</a> | <a href="https://raygauss.github.io/">Project Page</a>  
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="./media/Dex-NeRF_RayGauss_v2.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We present an enhanced differentiable ray-casting algorithm for rendering Gaussians with scene features, enabling efficient 3D scene learning from images.
</p>
<br>

## Hardware Requirements
  - CUDA-ready GPU
  - 24 GB VRAM (to train to paper evaluation quality)

## Software Requirements

The following software components are required to ensure compatibility and optimal performance:

- **Ubuntu**
- **NVIDIA Drivers**: Install NVIDIA drivers, version **525.60.13 or later**, to ensure compatibility with **CUDA Toolkit 12.4**, required in Conda environment setup.
- **NVIDIA OptiX 7.6**: NVIDIA’s OptiX ray tracing engine, version 7.6, is required for graphics rendering and computational tasks. You can download it from the [NVIDIA OptiX Legacy Downloads page](https://developer.nvidia.com/designworks/optix/downloads/legacy).
- **Anaconda**: Install [Anaconda](https://anaconda.com/download), a distribution that includes Conda, for managing packages and environments efficiently.

## Installation

Follow the steps below to set up the project:

   ```bash
  #Python-Optix requirements
  export OPTIX_PATH=/path/to/optix
  #For example if the repo is in your home folder: export OPTIX_PATH=~/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/
  export OPTIX_EMBED_HEADERS=1 # embed the optix headers into the package

  
  git clone https://github.com/hugobl1/ray_gauss.git
  cd ray_gauss
  conda env create --file environment.yml
  conda activate ray_gauss
  ```


# Dataset
### NeRF Synthetic Dataset
Please download and unzip [nerf_synthetic.zip](https://drive.google.com/file/d/1a3l9OL2lRA3z490QFNoDdZuUxTWrbdtD/view?usp=sharing). The folder contains initialization point clouds and the NeRF-Synthetic dataset.

#### Trained Models

If you would like to directly visualize a model trained by RayGauss, we provide the trained point clouds for each scene in NeRF-Synthetic. In this case, you can skip the training of the scene and evaluate or visualize it directly: [Download Link](https://drive.google.com/file/d/1E0_Tg2QeMx2kyohPhfRtfV656oQFQ2Kv/view?usp=sharing).


### Mip-NeRF 360 Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) website.

Place the datasets in the `dataset` folder.

# Training and Evaluation
To reproduce the results on entire datasets, follow the instructions below:

---

### NeRF-Synthetic Dataset
1. **Prepare the Dataset**: Ensure the NeRF-Synthetic dataset is downloaded and placed in the `dataset` directory.

2. **Run Training Script**: Execute the following command:

   ```bash
   bash nerf_synth.sh
    ```

This will start the training and evaluation on the NeRF-Synthetic dataset with the configuration parameter in `nerf_synthetic.yml`.

---

### Mip-NeRF 360 Dataset
To reproduce results on the **Mip-NeRF 360** dataset:

1. **Prepare the Dataset**: Download and place the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset in the `dataset` directory.

2. **Run Training Script**: Execute the following command:

   ```bash
   bash mip_nerf360.sh
    ```
---
3. **Results**: The results for each scene can be found in the `output` folder after training is complete.

### Single Scene
To train and test a single scene, simply use the following commands:

   ```bash
    python main_train.py -config "path_to_config_file" --save_dir "name_save_dir" --arg_names scene.source_path --arg_values "scene_path"
    python main_test.py -output "./output/name_save_dir" -iter save_iter
    # For example, to train and evaluate the hotdog scene from NeRF Synthetic:
    # python main_train.py -config "./configs/nerf_synthetic.yml" --save_dir "hotdog" --arg_names scene.source_path --arg_values "./dataset/nerf_synthetic/hotdog"
    # python main_test.py -output "./output/hotdog" -iter 30000
```


        
By default, only the last iteration is saved (30000 in the base config files).

# PLY Point Cloud Extraction
To extract a point cloud in PLY format from a trained scene, we provide the script [convertpth_to_ply.py](convertpth_to_ply.py), which can be used as follows:
   ```bash
   python convertpth_to_ply.py -output "./output/name_scene" -iter num_iter
   # For example, if the 'hotdog' scene was trained for 30000 iterations, you can use:
   # python convertpth_to_ply.py -output "./output/hotdog" -iter 30000
   ```

The generated PLY point cloud will be located in the folder `./output/scene/saved_pc/`.

# Visualization
To visualize a trained scene, we provide the script [main_gui.py](main_gui.py), which opens a GUI to display the trained scene:

   ```bash
   # Two ways to use the GUI:
   
   # Using the folder of the trained scene and the desired iteration
   python main_gui.py -output "./output/name_scene" -iter num_iter

   # Using a PLY point cloud:
   python main_gui.py -ply_path "path_to_ply_file"
   ```

## Camera Controls

### First Person Camera
In *First Person* mode, you can use the keyboard keys to move the camera in different directions.

- **Direction Keys**:
  - `Z`: Move forward
  - `Q`: Move backward
  - `S`: Move left
  - `D`: Move right
  - `A`: Move down
  - `E`: Move up  
  

- **View Control with Right Click**:
  - **Right Click + Move Mouse Up**: Look up
  - **Right Click + Move Mouse Down**: Look down
  - **Right Click + Move Mouse Left**: Look left
  - **Right Click + Move Mouse Right**: Look right

> **Note**: Ensure that the *First Person* camera mode is active for these controls to work.

### Trackball Camera
In *Trackball* mode, the camera can be controlled with the mouse to freely view around an object.

- **Left Click**: Rotate the camera around the object. Hold down the left mouse button and move the mouse to rotate around the object.
- **Right Click**: Pan. Hold down the right mouse button and move the mouse to shift the view laterally or vertically.
- **Mouse Wheel**: Zoom in and out. Scroll the wheel to adjust the camera's distance from the object.

> **Note**: Ensure that the *Trackball* camera mode is active for these controls to work.

# Camera Path Rendering

To render a camera path from a trained point cloud, use the script as follows:
```bash
python render_camera_path.py -output "./output" -camera_path_filename "camera_path.json" -name_video "my_video"
```
The `camera_path.json` file, which defines the camera path, can be generated using [NeRFStudio](https://nerf.studio/). 
This script loads a pre-trained model, renders images along a specified camera path, and saves them in `output/camera_path/images/`. A video is then generated from the images and saved in `output/camera_path/video/`.

# Using a Reality Capture Dataset
To use a dataset created with Reality Capture, refer to the [Reality Capture Instructions](./reality_capture/README.md).

# Acknowledgements

We thank the authors of [Python-Optix](https://github.com/mortacious/python-optix), upon which our project is based, as well as the authors of [NeRF](https://github.com/bmild/nerf) and [Mip-NeRF 360](https://github.com/google-research/multinerf) for providing their datasets. Finally, we would like to acknowledge the authors of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), as our project's dataloader is inspired by the one used in 3DGS.



# Citation
If you find our code or paper useful, please cite
```bibtex
@misc{blanc2024raygaussvolumetricgaussianbasedray,
      title={RayGauss: Volumetric Gaussian-Based Ray Casting for Photorealistic Novel View Synthesis}, 
      author={Hugo Blanc and Jean-Emmanuel Deschaud and Alexis Paljic},
      year={2024},
      eprint={2408.03356},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.03356}, 
}
```
