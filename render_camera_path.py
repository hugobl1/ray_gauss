import torch, argparse, os, json
import matplotlib.pyplot as plt
import numpy as np
from classes import point_cloud
from omegaconf import OmegaConf
from scripts import test
from utils.graphics_utils import focal2fov, fov2focal
from scripts.ply import write_ply
#Create a small perspective camera class

class PerspectiveCamera:
    def __init__(self,world_view_transform,camera_center,width,height,FoVx,FoVy,Px,Py,z_near,z_far):
        self.world_view_transform=world_view_transform
        # torch.linalg.inv(world_view_transform)
        self.camera_center=camera_center
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.Px = Px
        self.Py = Py
        self.z_near = z_near
        self.z_far = z_far
    def to(self,device):
        self.world_view_transform=self.world_view_transform.to(device)
        self.camera_center=self.camera_center.to(device)
        return self

def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length

parser = argparse.ArgumentParser(description="Ray Gauss Training.")
parser.add_argument("-output", type=str, default="output", help="Path to output folder")
parser.add_argument("-iter", type=int, default=-1, help="Test iteration")
parser.add_argument("-camera_path_filename", type=str, default="camera_path.json", help="Camera path filename")
parser.add_argument("-name_video", type=str, default="vid", help="Name of the video")
parser.add_argument("-keep_original_world_coordinates", type=int, default=1, help="Keep original world coordinates") #NerfStudio modified the world frame?
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

    config=OmegaConf.load(config_path)
    pointcloud = point_cloud.PointCloud(data_type="float32",device="cuda")
    #Load the model
    first_iter=pointcloud.restore_model(test_iter,path_model,config.training.optimization)
    camera_path_filename = args.camera_path_filename
    cam_list = []
    with open(camera_path_filename, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
        image_height = camera_path["render_height"]
        image_width = camera_path["render_width"]
        for camera in camera_path["camera_path"]:
            c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)
            if not args.keep_original_world_coordinates:
                c2w[2,:]*=-1.0
                c2w=c2w[np.array([0,2,1,3]),:]
            c2w[:3, 1:3] *= -1.0
            cam_center = c2w[:3, 3]
            # fovx = camera["fov"]
            # fovy = focal2fov(fov2focal(fovx, image_width), image_height)
            fov = camera["fov"]
            focal_length = three_js_perspective_camera_focal_length(fov, image_height)
            fovx=2*np.arctan(image_width/(2*focal_length))
            fovy=2*np.arctan(image_height/(2*focal_length))
            cam_list.append(PerspectiveCamera(c2w,cam_center,image_width,image_height,fovx,fovy,0.0,0.0,0.01,1000.0))
    #To device
    for i in range(len(cam_list)):
        cam_list[i]=cam_list[i].to("cuda")

    ############################################################################################################
    #############################Show camera positions##########################################################
    # #Save pointcloud in test_cam_path
    # xyz = pointcloud.positions.cpu().detach().numpy()
    # rgb = pointcloud.rgb.cpu().detach().numpy()
    # #Concatenate cam_center to xyz
    # cam_center = torch.stack([cam.camera_center for cam in cam_list],dim=0).cpu().detach().numpy()
    # xyz = np.concatenate([cam_center,cam_center],axis=0)
    # red=np.zeros((cam_center.shape[0],3))
    # red[:,0]=np.arange(cam_center.shape[0])/cam_center.shape[0]
    # rgb = np.concatenate([red,red],axis=0)
    
    # write_ply("./test_cam_path/pointcloud.ply", (xyz, rgb), ['x', 'y', 'z', 'red', 'green', 'blue'])
    # exit(0)
    ############################################################################################################
    print("cam_list",len(cam_list))
    # image_list = []
    # for cam in cam_list:
    #     im=test.render(pointcloud,cam,config.training.max_prim_slice,rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y), white_background=config.scene.white_background)
    #     image_list.append(im)
    image_list=test.render(pointcloud,cam_list,config.training.max_prim_slice,rnd_sample=config.training.rnd_sample,supersampling=(config.training.supersampling_x,config.training.supersampling_y), white_background=config.scene.white_background)
    ############################################################################################################
    #Save in os.path.join(path_output,"camera_path")
    path_cam_path=os.path.join(path_output,"camera_path")
    if not os.path.exists(path_cam_path):
        os.makedirs(path_cam_path)
    #Copy camera_path_filename to path_cam_path
    os.system("cp "+camera_path_filename+" "+path_cam_path)
    #Save images in path_cam_path/images
    path_images=os.path.join(path_cam_path,"images")
    if not os.path.exists(path_images):
        os.makedirs(path_images)
    for i in range(len(image_list)):
        #name_im is 6 digits
        num_im=str(i).zfill(6)
        # path_im="./test_cam_path/images/im"+num_im+".png"
        path_im=os.path.join(path_images,"im"+num_im+".png")
        plt.imsave(path_im,image_list[i])
    #Make a video from images with ffmpeg
    # ffmpeg -framerate 30 -i im%06d.png -c:v libx264 -q:v 1 -pix_fmt yuv420p libx264.mp4
    #Make a saved video folder in output
    path_video=os.path.join(path_cam_path,"video")
    if not os.path.exists(path_video):
        os.makedirs(path_video)
    os.system("ffmpeg -framerate 30 -i "+os.path.join(path_images,"im%06d.png")+" -c:v libx264 -q:v 1 -pix_fmt yuv420p "+os.path.join(path_video,args.name_video+".mp4"))