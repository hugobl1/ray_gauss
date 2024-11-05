# Launch main_train.py on every scene in mip_nerf360=["./dataset/mipnerf360/bycicle", "./dataset/mipnerf360/bonsai", "./dataset/mipnerf360/counter", "./dataset/mipnerf360/flowers", "./dataset/mipnerf360/garden", "./dataset/mipnerf360/kitchen", "./dataset/mipnerf360/room", "./dataset/mipnerf360/stump", "./dataset/mipnerf360/treehill"]
dataset_path="./dataset/mipnerf360/360_v2"
for scene in bicycle bonsai counter flowers garden kitchen room stump treehill
do
    scene_path="${dataset_path}/${scene}"
    python main_train.py -config "configs/mip_nerf.yml" --save_dir "${scene}" --arg_names scene.source_path --arg_values "${scene_path}"
    output_path="output/${scene}"
    python main_test.py -output "${output_path}" -iter 30000
done
