# Tutorial to create a COLMAP dataset from Reality Capture

## Reality Capture

Reality Capture is a professional photogrammetry software solution (available for Windows, Linux, and MacOS), allowing a faster and more accurate pose estimation than COLMAP. This software is completely free for academics. Here are the steps to generate a dataset in COLMAP format for training with our RayGauss method (and also 3D Gaussian Splatting and variants).

Thanks to Jonathan Chemla (CTO at Iconem) for the solution: https://github.com/nerfstudio-project/nerfstudio/issues/2419

The solution is to export the data from the Reality Capture software in bundler format, then with the Python kapture library to export it in COLMAP format.

In Reality Capture, you only need to do the "ALIGNMENT" step to calculate the poses, the intrinsic parameters of the cameras and the sparse point cloud. With the "Export" button, you choose to export the data in "Bundler v0.3" format.

Here are the export settings to create undistorted images and parameters in bundler format:

![export_RC](https://github.com/user-attachments/assets/566eed29-83b1-4217-a1c0-27ad84fa199f)

It is important to set the scales to 0.1 because the RayGauss optimization hyperparameters are chosen for a standard COLMAP output scale.

Then put all the images in an `images` directory.

You need to export also a file named "imagelist-local.lst" that contains the list of images. You can get this file by doing "Export" -> "Undistorted images with imageslist" in Reality Capture.

The final data should be assembled in this form inside the folder `dataset-bundler`:

![export_RC_2](https://github.com/user-attachments/assets/24d68387-12e8-4df9-83a7-883d3a217ba9)


## The different steps once the data is exported from Reality Capture

1. **Install Dependencies in your conda environment (only first time)**
   ```bash
   pip install kapture
   ```

2. **Reconstruct Dataset with Kapture**  
   ```bash
   python /path/to/your/conda/env\Scripts\kapture_import_bundler.py -v debug -i dataset-bundler\bundle.out -l dataset-bundler\imagelist-local.lst -im dataset-bundler\images --image_transfer link_absolute -o dataset-kapture --add-reconstruction
   #For example: python C:\Users\JE\miniconda3\envs\python39-env\Scripts\kapture_import_bundler.py -v debug -i dataset-bundler\bundle.out -l dataset-bundler\imagelist-local.lst -im dataset-bundler\images --image_transfer link_absolute -o dataset-kapture --add-reconstruction
   ```

3. **Crop borders of the images (optional)**  
   ```bash
   python kapture-cropper.py -v info -i dataset-kapture\ --border_px 10
   ```

4. **Export the dataset to COLMAP format**  
    - If you have cropped the images then copy the `dataset-kapture\sensors\records_data` directory into `dataset-colmap\` and rename the directory as `images`
    - Otherwise copy the `images` directory from `dataset-bundler\` to `dataset-colmap\`
    - Then, run the following script:
   ```bash
   python /path/to/your/conda/env\Scripts\kapture_export_colmap.py -v debug -f -i dataset-kapture -db dataset-colmap\colmap.db --reconstruction dataset-colmap\reconstruction-txt
   #For example: python C:\Users\JE\miniconda3\envs\python39-env\Scripts\kapture_export_colmap.py -v debug -f -i dataset-kapture -db dataset-colmap\colmap.db --reconstruction dataset-colmap\reconstruction-txt
   ```

5. **Prepare the COLMAP Sparse Directory**
   - Create the `sparse` folder in `dataset-colmap\`
   - Create the folder `0` within `dataset-colmap\sparse\`.

7. **Convert the Model Format (you need to add the COLMAP folder in your PATH env in Windows)**  
   ```bash
   COLMAP.bat model_converter --input_path dataset-colmap\reconstruction-txt --output_path dataset-colmap\sparse\0 --output_type BIN
   ```

8. **Generate the point cloud in PLY format (by default, the script keeps only points visible from at least 4 cameras)**  
   ```bash
   python create_ply_from_reconstruction.py
    ```

   You can now train RayGauss with your new dataset.
