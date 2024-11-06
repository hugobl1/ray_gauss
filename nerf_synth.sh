# Launch main_train.py on every scene in nerf_synthetic=["./nerf_synthetic/mic","./nerf_synthetic/chair","./nerf_synthetic/ship","./nerf_synthetic/materials","./nerf_synthetic/lego","./nerf_synthetic/drums","./nerf_synthetic/ficus","./nerf_synthetic/hotdog"]
dataset_path="./dataset/nerf_synthetic"
for scene in chair drums ficus hotdog lego materials mic ship
do
    scene_path="${dataset_path}/${scene}"
    ply_name="fused_light.ply"
    python main_train.py -config "configs/nerf_synthetic.yml" --save_dir "${scene}" --arg_names scene.source_path pointcloud.ply.ply_name --arg_values "${scene_path}" "${ply_name}"
    output_path="output/${scene}"
    python main_test.py -output "${output_path}" -iter 30000
done