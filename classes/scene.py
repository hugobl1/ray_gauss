import torch
import os
# import nerfacc
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

from classes.cameras import create_scaled_cameras

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON,cameraList_from_camInfos_native_multi_scale
from utils.system_utils import searchForMaxIteration
import random
# import json

from classes import point_cloud

class Scene:
    "Scene"

    def __init__(self, config , pointcloud : point_cloud.PointCloud, shuffle=False, train_resolution_scales=[1.0], test_resolution_scales=[1.0],init_pc=True):

        self.model_path = config.save.models
        self.pointcloud = pointcloud

        #Si la méthode d'initialisation est "ply", ajoute une clé "ply_path" à la configuration
        if (config.pointcloud.init_method == "ply"):
            config.pointcloud.ply.ply_path = os.path.join(config.scene.source_path, config.pointcloud.ply.ply_name)

        #Make sure the path is absolute
        config.scene.source_path = os.path.abspath(config.scene.source_path)

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(config.scene.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](config.scene.source_path, config.scene.images, config.scene.eval,config)
        elif os.path.exists(os.path.join(config.scene.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](config.scene.source_path, config.scene.white_background, config.scene.eval,config)
        elif os.path.exists(os.path.join(config.scene.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](config.scene.source_path,config.pointcloud.ply.ply_name,config.scene.white_background, config.scene.eval, config.scene.load_allres)
        elif config.scene.source_path == "/":
            # custom test scene
            return
        else:
            assert False, "Could not recognize scene type!"

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in train_resolution_scales:
            if isinstance(scene_info.train_cameras, dict): #if scene_info.train_cameras is a dictionary, then it is a native multi-scale dataset
                if resolution_scale in scene_info.train_cameras.keys():
                    self.train_cameras[resolution_scale] = cameraList_from_camInfos_native_multi_scale(scene_info.train_cameras[resolution_scale], config.scene)
                else:
                    print("No train cameras for resolution scale",resolution_scale)
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, config.scene)

        for resolution_scale in test_resolution_scales:
            if isinstance(scene_info.test_cameras, dict): #if scene_info.test_cameras is a dictionary, then it is a native multi-scale dataset 
                if resolution_scale in scene_info.test_cameras.keys():
                    self.test_cameras[resolution_scale] = cameraList_from_camInfos_native_multi_scale(scene_info.test_cameras[resolution_scale], config.scene)
                else:
                    print("No test cameras for resolution scale",resolution_scale)
            else:
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, config.scene)

        self.train_resolution_scales=self.train_cameras.keys()
        self.test_resolution_scales=self.test_cameras.keys()
        self.highest_training_scale = min(self.train_cameras.keys())
        self.highest_testing_scale = min(self.test_cameras.keys())
        
        if init_pc:
            self.pointcloud.init_cloud_attributes(config.pointcloud)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.pointcloud.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def scaleTrainCameras(self, scaling_factor, scale_list=None):
        scale_list = self.train_resolution_scales if scale_list is None else scale_list
        for scale in scale_list:
            new_train_cameras = []
            for i in range(len(self.train_cameras[scale])):
                cam=self.train_cameras[scale][i]
                new_train_cameras.append(create_scaled_cameras(cam,scaling_factor))
            self.train_cameras[scale] = new_train_cameras

    
    def scaleTestCameras(self, scaling_factor, scale_list=None):
        scale_list = self.test_resolution_scales if scale_list is None else scale_list
        for scale in scale_list:
            new_test_cameras = []
            for i in range(len(self.test_cameras[scale])):
                cam=self.test_cameras[scale][i]
                new_test_cameras.append(create_scaled_cameras(cam,scaling_factor))
            self.test_cameras[scale] = new_test_cameras

    def getTrainCameras(self, scale=None):
        if scale is None:
            scale = [self.highest_training_scale]
        cam_list=[]
        for s in scale:
            cam_list.extend(self.train_cameras[s])
        return cam_list

    def getTestCameras(self, scale=None):
        if scale is None:
            scale = [self.highest_testing_scale]
        cam_list=[]
        for s in scale:
            cam_list.extend(self.test_cameras[s])        
        return cam_list