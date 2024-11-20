import logging
logging.basicConfig(
    filename='log_output.txt',  # Chemin du fichier
    level=logging.INFO,        # Niveau minimal de journalisation
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format des messages
)
import argparse, os
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from classes import point_cloud,scene
from scripts import test
from utils.loss_utils import ssim
from utils.metrics_utils import PSNR
from lpipsPyTorch import lpips

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Ray Gauss Training.")
parser.add_argument("-output", type=str, default="output", help="Path to output folder")
parser.add_argument("-iter", type=int, default=-1, help="Test iteration")
args = parser.parse_args()

if __name__ == "__main__":
    ############################################################################################################
    path_output = args.output
    #Check if output folder exists
    if not os.path.exists(path_output):
        print("Output folder not found")
        exit()
    path_model = os.path.join(path_output, "models")
    #Check if test_iter==-1
    if args.iter==-1:
        #Get the last iteration you can found in model folder "chkpnt${iter}.pth"
        list_files=os.listdir(path_model)
        list_files.sort()
        last_iter=-1
        for file in list_files:
            if "chkpnt" in file:
                last_iter=int(file.split("chkpnt")[1].split(".pth")[0])
        if last_iter==-1:
            print("No checkpoint found")
            exit()
        else:
            test_iter=last_iter
    else:
        test_iter=args.iter

    config_path = os.path.join(path_output, "config",
                            os.listdir(os.path.join(path_output, "config"))[0])
    ############################################################################################################
    #Create test_results folder
    path_test_results=os.path.join(path_output,"test_results")
    os.makedirs(os.path.join(path_test_results,"renders"),exist_ok=True)
    os.makedirs(os.path.join(path_test_results,"gts"),exist_ok=True)
    ############################################################################################################

    config=OmegaConf.load(config_path)
    if config.scene.eval==False:
        print("This scene was trained with no test data: eval==False")
        exit()
    test_point_cloud=point_cloud.PointCloud(data_type=config.pointcloud.data_type,device=device)
    tested_scene=scene.Scene(config=config,pointcloud=test_point_cloud,train_resolution_scales=config.scene.train_resolution_scales,test_resolution_scales=config.scene.test_resolution_scales,init_pc=False)
    #Load the model
    first_iter=tested_scene.pointcloud.restore_model(test_iter,path_model,config.training.optimization)

    tested_cams=tested_scene.getTestCameras()
    PSNR_list, gt_images_list, images_list=test.inference(tested_scene.pointcloud,tested_cams,config.training.max_prim_slice,rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y), white_background=config.scene.white_background)
    #Save gt_images_list inside test_results/gts and images_list inside test_results/renders
    PSNR_list=[]
    SSIM_list=[]
    LPIPS_list=[]
    for i in tqdm(range(len(gt_images_list))):
        path_gt=os.path.join(path_test_results,"gts","gt"+str(i)+".png")
        path_render=os.path.join(path_test_results,"renders","render"+str(i)+".png")
        plt.imsave(path_gt,gt_images_list[i])
        plt.imsave(path_render,images_list[i])
        PSNR_list.append(PSNR(images_list[i],gt_images_list[i]))
        permute_images=torch.tensor(images_list[i].transpose(2,0,1),dtype=torch.float32,device="cuda")[None,...]
        permute_gt_images=torch.tensor(gt_images_list[i].transpose(2,0,1),dtype=torch.float32,device="cuda")[None,...]
        SSIM_list.append(ssim(permute_images,permute_gt_images).item())
        LPIPS_list.append(lpips(permute_images,permute_gt_images, net_type='vgg').item())

    with open(os.path.join(path_test_results,"metrics.txt"),"w") as f:
        f.write("Mean PSNR SSIM LPIPS\n")
        mean_PSNR=sum(PSNR_list)/len(PSNR_list)
        mean_SSIM=sum(SSIM_list)/len(SSIM_list)
        mean_LPIPS=sum(LPIPS_list)/len(LPIPS_list)
        f.write(str(mean_PSNR)+" "+str(mean_SSIM)+" "+str(mean_LPIPS)+"\n")
        f.write("Iter PSNR SSIM LPIPS\n")
        for i in range(len(PSNR_list)):
            f.write(str(test_iter)+" "+str(PSNR_list[i])+" "+str(SSIM_list[i])+" "+str(LPIPS_list[i])+"\n")


    