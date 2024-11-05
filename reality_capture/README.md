# Setup Instructions

1. **Install Dependencies**  
   ```bash
   pip install kapture
   ```

2. **Activate the Conda Environment**  
   ```bash
   conda activate python39-env
   ```

3. **Import Dataset with Kapture**  
   ```bash
   python C:\Users\JE\miniconda3\envs\python39-env\Scripts\kapture_import_bundler.py -v debug -i dataset-bundler\bundle.out -l dataset-bundler\imagelist-local.lst -im dataset-bundler\images --image_transfer link_absolute -o dataset-kapture --add-reconstruction
   ```

4. **Crop Dataset Borders (Optional)**  
   ```bash
   python kapture-cropper.py -v info -i dataset-kapture\ --border_px 10
   ```

5. **Export Dataset to COLMAP**  
   ```bash
   python C:\Users\JE\miniconda3\envs\python39-env\Scripts\kapture_export_colmap.py -v debug -f -i dataset-kapture -db dataset-colmap\colmap.db --reconstruction dataset-colmap\reconstruction-txt`
   ```

6. **Prepare the COLMAP Sparse Directory**  
   - Create the folder `0` within `dataset-colmap\sparse\`.

7. **Convert the Model Format**  
   ```bash
   COLMAP.bat model_converter --input_path dataset-colmap\reconstruction-txt --output_path dataset-colmap\sparse\0 --output_type BIN
   ```

8. **Generate PLY from Reconstruction**  
   ```bash
   python create_ply_from_reconstruction.py
    ```