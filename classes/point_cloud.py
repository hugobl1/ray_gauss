import torch
import scripts.ply as ply
import numpy as np
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from classes import optimizer
from utils.graphics_utils import fov2focal
from utils.sh_utils import RGB2SH, SH2RGB
from torch import nn
from simple_knn._C import distCUDA2
from utils.optim_utils import define_optimizer_manager
from utils.general_utils import inverse_sigmoid,build_rotation

import math


class PointCloud:
  """Point Cloud class.
    """
  def __init__(self, data_type,device):
    print("#"*80)
    print("PointCloud initialization")
    self.set_data_type(data_type)
    self.display_data_type()
    self.device=device
    
    self.positions=torch.empty(0)
    self.rgb=torch.empty(0)

    # self.harmonic_number = config.harmonic_number
    self.spherical_harmonics=torch.empty(0)
    self.densities=torch.empty(0)
    self.scales=torch.empty(0)
    self.quaternions=torch.empty(0)
    self.xyz_gradient_accum_norm=torch.empty(0)
    self.num_accum=torch.empty(0)
    self.num_accum_gnv=torch.empty(0)
    self.filter_3D=torch.empty(0)

    # self.num_sph_gauss=config.num_sph_gauss
    self.sph_gauss_features=torch.empty(0)
    self.bandwidth_sharpness=torch.empty(0)
    self.lobe_axis=torch.empty(0)

    # self.init_cloud_attributes(config)

    ###############################################################################################
  def set_data_type(self,data_type):
    if(data_type=="float32"):
      self.data_type=torch.float32
    elif(data_type=="float64"):
      self.data_type=torch.float64
    else:
      print("Error: data_type must be float32 or float64")
      sys.exit()

  def display_data_type(self):
    print("PointCloud: data type: ",self.data_type)
    return

  def  init_cloud_attributes(self,config):
    """ Initialize the attributes of the pointcloud
    Parameters:
    -----------
    init_params: dict
      Dictionary containing the attributes of the pointcloud

    """
    type=config.init_method
    if type=="ply":
        cfg=config.ply
        self.load_from_ply(cfg.ply_path,cfg.sh_nb,cfg.sg_nb,cfg.default_density)
    elif type=="rg_ply":
        cfg=config.rg_ply
        self.load_from_rg_ply(cfg.path)
    elif type=="pt":
        cfg=config.pt
        self.init_from_pt(cfg.iter,cfg.model_folder)
    elif type=="pth":
        cfg=config.pth
        self.init_from_pth(cfg.iter,cfg.model_folder)
    elif type=="gaussian_splatting":
        cfg=config.gaussian_splatting
        self.init_from_gaussian_splatting(config)
    elif type=="uniform":
        cfg=config.ply
        # self.load_from_ply(cfg.ply_path,cfg.sh_nb,cfg.sg_nb,cfg.default_density, True)
        self.init_uniform(cfg.sh_nb,cfg.sg_nb,config.uniform.n_rnd_pts)
        
    # elif type=="random":
    #     cfg=config.random
    #     self.init_random(cfg.num_pts,cfg.sh_nb,cfg.sg_nb,
    #                      cfg.default_density)
    else:
        raise ValueError("Unknown type of pointcloud")

  def register_config(self,config,folder,iter):
    config.pointcloud={}
    config.pointcloud.harmonic_number=self.harmonic_number
    config.pointcloud.init_method="pt"
    config.pointcloud.ply={}
    config.pointcloud.ply.path=""
    config.pointcloud.data_type="float32" if self.data_type==torch.float32 else "float64"
    config.pointcloud.pt={}
    config.pointcloud.pt.model_folder=folder
    config.pointcloud.pt.iter=iter
    return config
  
  def init_from_gaussian_splatting(self,init_params):
    self.load_from_gaussian_splatting(name=init_params.gaussian_splatting.path,num_harmonics=init_params.harmonic_number)
  
  def init_uniform(self, sh_nb, sg_nb, num_pts: int = 100_000):
    """for a comparative baseline, init randomly

    Args:
        num_pts (int): _description_
    """
    self.harmonic_number = sh_nb
    self.num_sph_gauss   = sg_nb
    
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    # shs = np.random.random((num_pts, 3)) / 255.0
    
    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().cuda()), 0.0000001)
    
    self.positions = torch.from_numpy(xyz).to(dtype=self.data_type, device=device).contiguous().requires_grad_(True)
    # self.spherical_harmonics=torch.load(folder_tensors+'/'+name_spherical_harmonics+'.pt')#[:,:,:self.harmonic_number]
    
    # self.rgb=color_features[:,:,0]
    # self.spherical_harmonics=color_features[:,:,1:]
    colors = np.zeros((num_pts, 3))
    
    with torch.no_grad():
      self.rgb=torch.tensor(colors*2*np.sqrt(np.pi),device=device,dtype=self.data_type).contiguous().requires_grad_(True)
    self.rgb.requires_grad_(True)
    
    self.spherical_harmonics = torch.zeros((num_pts,3,self.harmonic_number-1),device=device,dtype=self.data_type, requires_grad=True).contiguous()
    
    # 10 is default density
    self.densities = torch.full( (num_pts,), 10, dtype=self.data_type, device=device, requires_grad=True ).contiguous()
    with torch.no_grad():
      self.scales = torch.log( torch.sqrt(dist2)[..., None].repeat(1, 3) )
    self.scales.contiguous().requires_grad_(True)
    
    self.quaternions = torch.zeros( (num_pts, 4), dtype=self.data_type, device=device, requires_grad=True )
    with torch.no_grad():
      self.quaternions[:,0] = 1.0
    self.quaternions.contiguous().requires_grad_(True)

    self.sph_gauss_features=torch.zeros((num_pts,3,self.num_sph_gauss),dtype=self.data_type,device=device,requires_grad=True)
    self.bandwidth_sharpness=torch.zeros((num_pts,self.num_sph_gauss),dtype=self.data_type,device=device,requires_grad=True)
    self.lobe_axis=torch.zeros((num_pts,self.num_sph_gauss,3),dtype=self.data_type,device=device,requires_grad=True)
    
    self.xyz_gradient_accum_norm = torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum = torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum_gnv = torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    
    filter_3D = np.zeros(num_pts)[..., np.newaxis] + 0.0001
    self.filter_3D = torch.log(torch.from_numpy(filter_3D).type(torch.float32).to(device))
    
  
  def init_from_pt(self,iter,model_folder):
    """ Initialize the attributes of the pointcloud from a pt file

    Parameters:
    -----------
    iter: int
        Iteration of the model to load
    model_folder: str
        Folder containing the model
    """
    print("PointCloud initialized with a pt file")
    str_iter=str(iter)
    name_positions="positions_iter"+str_iter
    name_spherical_harmonics="spherical_harmonics_iter"+str_iter
    name_densities="densities_iter"+str_iter
    name_scales="scales_iter"+str_iter
    name_quaternions="quaternions_iter"+str_iter

    name_sph_gauss_features="sph_gauss_features_iter"+str_iter
    name_bandwidth_sharpness="bandwidth_sharpness_iter"+str_iter
    name_lobe_axis="lobe_axis_iter"+str_iter
    
    self.load_from_pt(name_positions,name_spherical_harmonics,name_densities, name_scales,name_quaternions,
                      name_sph_gauss_features,name_bandwidth_sharpness,name_lobe_axis,
                      folder_tensors=model_folder)
    
  def init_from_pth(self,iter,model_folder):
    """ Initialize the attributes of the pointcloud from a pth file

    Parameters:
    -----------
    iter: int
        Iteration of the model to load
    model_folder: str
        Folder containing the model

    """
    print("PointCloud initialized with a pth file")
    self.restore_model(iter,model_folder)
    
  def setup_optimizers(self,optimizer_dict):
    self.optim_managers=optimizer.OptimizerManagerDict(optimizer_dict)

  def save_pt_epoch(self,epoch,folder_tensors="saved_tensors",prefix=""):
    """Save the pointcloud, spherical_harmonics and densities in a folder with pt files."""
    name_positions=prefix+"positions_iter"+str(epoch)
    name_spherical_harmonics=prefix+"spherical_harmonics_iter"+str(epoch)
    name_densities=prefix+"densities_iter"+str(epoch)
    name_scales=prefix+"scales_iter"+str(epoch)
    name_quaternions=prefix+"quaternions_iter"+str(epoch)
    name_sph_gauss_features=prefix+"sph_gauss_features_iter"+str(epoch)
    name_bandwidth_sharpness=prefix+"bandwidth_sharpness_iter"+str(epoch)
    name_lobe_axis=prefix+"lobe_axis_iter"+str(epoch)
    self.save_pt(name_positions=name_positions,name_spherical_harmonics=name_spherical_harmonics,name_densities=name_densities, name_scales=name_scales, name_quaternions=name_quaternions,
              name_sph_gauss_features=name_sph_gauss_features,name_bandwidth_sharpness=name_bandwidth_sharpness,name_lobe_axis=name_lobe_axis,
              folder_tensors=folder_tensors)

  def save_pt(self,name_positions="positions",name_spherical_harmonics="spherical_harmonics",name_densities="densities", name_scales="scales", name_quaternions="quaternions",
              name_sph_gauss_features="sph_gauss_features",name_bandwidth_sharpness="bandwidth_sharpness",name_lobe_axis="lobe_axis",
              folder_tensors="saved_tensors"):
    """Save the pointcloud, spherical_harmonics and densities in a folder with pt files."""
    torch.save(self.positions,folder_tensors+'/'+name_positions+'.pt')
    color_features=torch.cat([self.rgb[:,:,None],self.spherical_harmonics],dim=2)
    torch.save(color_features,folder_tensors+"/"+name_spherical_harmonics+".pt")
    torch.save(self.densities,folder_tensors+"/"+name_densities+".pt")
    torch.save(self.scales,folder_tensors+"/"+name_scales+".pt")
    torch.save(self.quaternions,folder_tensors+"/"+name_quaternions+".pt")
    torch.save(self.sph_gauss_features,folder_tensors+"/"+name_sph_gauss_features+".pt")
    torch.save(self.bandwidth_sharpness,folder_tensors+"/"+name_bandwidth_sharpness+".pt")
    torch.save(self.lobe_axis,folder_tensors+"/"+name_lobe_axis+".pt")
    xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    xyz_gradient_accum_norm[self.num_accum>0]=self.xyz_gradient_accum_norm[self.num_accum>0]/self.num_accum[self.num_accum>0]
    torch.save(xyz_gradient_accum_norm,folder_tensors+"/"+name_positions+"_xyz_gradient_accum_norm.pt")

  def load_from_pt(self,name_positions="positions",name_spherical_harmonics="spherical_harmonics",name_densities="densities", name_scales="scales", name_quaternions="quaternions",
                    name_sph_gauss_features="sph_gauss_features",name_bandwidth_sharpness="bandwidth_sharpness",name_lobe_axis="lobe_axis",
                   folder_tensors="saved_tensors"):
    """Load the pointcloud, spherical_harmonics and densities from the folder folder_tensors in the current object."""
    self.positions=torch.load(folder_tensors+'/'+name_positions+'.pt',weights_only=False)
    # self.spherical_harmonics=torch.load(folder_tensors+'/'+name_spherical_harmonics+'.pt')#[:,:,:self.harmonic_number]
    color_features=torch.load(folder_tensors+'/'+name_spherical_harmonics+'.pt',weights_only=False)
    # self.rgb=color_features[:,:,0]
    # self.spherical_harmonics=color_features[:,:,1:]
    self.rgb=torch.tensor(color_features[:,:,0].clone(),device=device,dtype=self.data_type).contiguous().requires_grad_(True)
    self.spherical_harmonics=torch.tensor(color_features[:,:,1:].clone(),requires_grad=True,device=device,dtype=self.data_type).contiguous().requires_grad_(True)
    self.harmonic_number=self.spherical_harmonics.shape[2]+1
    
    self.densities=torch.load(folder_tensors+"/"+name_densities+".pt",weights_only=False)
    self.scales=torch.load(folder_tensors+"/"+name_scales+".pt",weights_only=False)
    self.quaternions=torch.load(folder_tensors+"/"+name_quaternions+".pt",weights_only=False)

    self.sph_gauss_features=torch.load(folder_tensors+"/"+name_sph_gauss_features+".pt",weights_only=False)
    self.bandwidth_sharpness=torch.load(folder_tensors+"/"+name_bandwidth_sharpness+".pt",weights_only=False)
    self.lobe_axis=torch.load(folder_tensors+"/"+name_lobe_axis+".pt",weights_only=False)
    self.num_sph_gauss=self.sph_gauss_features.shape[2]
    
    self.xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.filter_3D = torch.log(torch.sqrt(torch.clamp_min(distCUDA2(self.positions), 0.0000001))[...,None].repeat(1, 3))
    
  def load_from_rg_ply(self,name):
    "Load a pointcloud from a ply file that was generated by raygauss (ours) program"
    print("Loading pointcloud from raygauss ply file:",name)
    pointcloud= ply.read_ply(name)
    # Check if the pointcloud has an x,y,z field
    fields=pointcloud.dtype.names
    if 'x' in fields and 'y' in fields and 'z' in fields:
      print("x,y,z coordinates found")
      points = np.vstack((pointcloud['x'], pointcloud['y'], pointcloud['z'])).T
      size_pointcloud=points.shape[0]
    else:
      print("No x,y,z coordinates in the pointcloud")
      sys.exit(0)
    # Check if the pointcloud has a density field
    if 'density' in fields:
      densities = pointcloud['density'].astype(np.float64)
    else:
      print("No density field in the pointcloud")
      exit(0)
    self.positions=torch.from_numpy(points).type(torch.float32).to(device).contiguous().requires_grad_(True)
    self.densities=torch.tensor(densities,requires_grad=True,device=device,dtype=self.data_type)
    # Check if the pointcloud has sh0r,sh0g,sh0b,...,shnr,shng,shnb and spherical gaussian features
    sh_or_sg=False
    n_sh=0
    name_shr,name_shg,name_shb="sh"+str(n_sh)+"r","sh"+str(n_sh)+"g","sh"+str(n_sh)+"b"
    while name_shr in fields and name_shg in fields and name_shb in fields:
      sh_or_sg=True
      n_sh+=1
      name_shr="sh"+str(n_sh)+"r"
      name_shg="sh"+str(n_sh)+"g"
      name_shb="sh"+str(n_sh)+"b"
    
    n_sg=0
    name_sgr,name_sgg,name_sgb="sg"+str(n_sg)+"r","sg"+str(n_sg)+"g","sg"+str(n_sg)+"b"
    while name_sgr in fields and name_sgg in fields and name_sgb in fields:
      sh_or_sg=True
      n_sg+=1
      name_sgr="sg"+str(n_sg)+"r"
      name_sgg="sg"+str(n_sg)+"g"
      name_sgb="sg"+str(n_sg)+"b"
    if not(sh_or_sg):
      print("No color field in the pointcloud")
      sys.exit(0)
    self.harmonic_number=n_sh
    self.num_sph_gauss=n_sg
    spherical_harmonics = np.zeros((size_pointcloud,3,n_sh))
    sph_gauss_features = np.zeros((size_pointcloud,3,n_sg))
    bandwidth_sharpness = np.zeros((size_pointcloud,n_sg))
    lobe_axis = np.zeros((size_pointcloud,n_sg,3))
    for i in range(n_sh):
      name_r="sh"+str(i)+"r"
      name_g="sh"+str(i)+"g"
      name_b="sh"+str(i)+"b"
      spherical_harmonics[:,:,i]=np.vstack((pointcloud[name_r], pointcloud[name_g], pointcloud[name_b])).T
    for i in range(n_sg):
      name_r,name_g,name_b="sg"+str(i)+"r","sg"+str(i)+"g","sg"+str(i)+"b"
      name_lx,name_ly,name_lz="lobe_"+str(i)+"x","lobe_"+str(i)+"y","lobe_"+str(i)+"z"
      name_bs="bandwidth_sharpness"+str(i)
      sph_gauss_features[:,:,i]=np.vstack((pointcloud[name_r], pointcloud[name_g], pointcloud[name_b])).T
      lobe_axis[:,i,:]=np.vstack((pointcloud[name_lx], pointcloud[name_ly], pointcloud[name_lz])).T
      bandwidth_sharpness[:,i]=pointcloud[name_bs]
    self.rgb=torch.tensor(spherical_harmonics[:,:,0],device=device,dtype=self.data_type).contiguous().requires_grad_(True)
    self.spherical_harmonics=torch.tensor(spherical_harmonics[:,:,1:],requires_grad=True,device=device,dtype=self.data_type)
    self.sph_gauss_features=torch.tensor(sph_gauss_features,requires_grad=True,device=device,dtype=self.data_type)
    self.bandwidth_sharpness=torch.tensor(bandwidth_sharpness,requires_grad=True,device=device,dtype=self.data_type)
    self.lobe_axis=torch.tensor(lobe_axis,requires_grad=True,device=device,dtype=self.data_type)

    if 'sx' in fields and 'sy' in fields and 'sz' in fields:
      print("Scale field found")
      scales=np.vstack((pointcloud['sx'], pointcloud['sy'], pointcloud['sz'])).T
      self.scales=torch.tensor(scales,device=device,dtype=self.data_type).contiguous().requires_grad_(True)
      self.scales.contiguous().requires_grad_(True)
    else:  
      print("No scale field in the raygauss ply file")
      sys.exit(0)
    if 'qx' in fields and 'qy' in fields and 'qz' in fields and 'qw' in fields:
      print("Rotation field found")
      quaternions = np.vstack((pointcloud['qx'], pointcloud['qy'], pointcloud['qz'], pointcloud['qw'])).T
      self.quaternions=torch.tensor(quaternions,device=device,dtype=self.data_type).contiguous().requires_grad_(True)
    else:
      print("No rotation field in the raygauss ply file")
      sys.exit(0)
    
    self.xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)

    filter_3D = np.zeros(points.shape[0])[..., np.newaxis] + 0.0001
    self.filter_3D = torch.log(torch.from_numpy(filter_3D).type(torch.float32).to(device))

  def load_from_ply(self,name,sh_nb,sg_nb,default_density, baseline=False): #default_spherical_harmonics=np.ones((3,6))*0.4*np.sqrt(np.pi),
    """Load a pointcloud from a ply file."""
    print("Loading pointcloud from ply file:",name)
    self.harmonic_number=sh_nb
    self.num_sph_gauss=sg_nb
    pointcloud= ply.read_ply(name)
    # Check if the pointcloud has an x,y,z field
    fields=pointcloud.dtype.names
    if 'x' in fields and 'y' in fields and 'z' in fields:
      print("x,y,z coordinates found")
      points = np.vstack((pointcloud['x'], pointcloud['y'], pointcloud['z'])).T
      size_pointcloud=points.shape[0]
    else:
      print("No x,y,z coordinates in the pointcloud")
      sys.exit(0)
    # Check if the pointcloud has a density field
    if 'density' in fields:
      print("Density field found")
      densities = pointcloud['density'].astype(np.float64)
    else:
      print("No density field in the pointcloud")
      densities = np.ones(size_pointcloud)*default_density
    self.positions=torch.from_numpy(points).type(torch.float32).to(device)
    self.positions=self.positions.contiguous()
    self.positions.requires_grad_(True)

    self.densities=torch.tensor(densities,requires_grad=True,device=device,dtype=self.data_type)
    # Check if the pointcloud has a color field
    if 'red' in fields and 'green' in fields and 'blue' in fields and not baseline: #Suppose only rgb
      print("RGB field found")
      colors = np.vstack((pointcloud['red'], pointcloud['green'], pointcloud['blue'])).T/255-0.5
      # Convert to  spherical harmonics
      spherical_harmonics = np.zeros((size_pointcloud,3,self.harmonic_number-1))
      self.rgb=torch.tensor(colors*2*np.sqrt(np.pi),device=device,dtype=self.data_type).contiguous().requires_grad_(True)
      self.spherical_harmonics=torch.tensor(spherical_harmonics,requires_grad=True,device=device,dtype=self.data_type)
    
      self.sph_gauss_features=torch.zeros((size_pointcloud,3,self.num_sph_gauss),dtype=self.data_type,device=device,requires_grad=True)
      self.bandwidth_sharpness=torch.zeros((size_pointcloud,self.num_sph_gauss),dtype=self.data_type,device=device,requires_grad=True)
      self.lobe_axis=torch.zeros((size_pointcloud,self.num_sph_gauss,3),dtype=self.data_type,device=device,requires_grad=True)
    else:
      
      colors = np.zeros((size_pointcloud, 3))
    
      with torch.no_grad():
        self.rgb=torch.tensor(colors*2*np.sqrt(np.pi),device=device,dtype=self.data_type).contiguous().requires_grad_(True)
      self.rgb.requires_grad_(True)
      
      self.spherical_harmonics = torch.zeros((size_pointcloud,3,self.harmonic_number-1),device=device,dtype=self.data_type, requires_grad=True).contiguous()

      self.sph_gauss_features=torch.zeros((size_pointcloud,3,self.num_sph_gauss),dtype=self.data_type,device=device,requires_grad=True)
      self.bandwidth_sharpness=torch.zeros((size_pointcloud,self.num_sph_gauss),dtype=self.data_type,device=device,requires_grad=True)
      self.lobe_axis=torch.zeros((size_pointcloud,self.num_sph_gauss,3),dtype=self.data_type,device=device,requires_grad=True)
    
      #Check if the pointcloud has sh0r,sh0g,sh0b,...,shnr,shng,shnb and spherical gaussian features
      # colors=np.zeros((size_pointcloud,3))
      # spherical_harmonics = np.zeros((size_pointcloud,3,self.harmonic_number-1))
      # sph_gauss_features = np.zeros((size_pointcloud,3,self.num_sph_gauss))
      # bandwidth_sharpness = np.zeros((size_pointcloud,self.num_sph_gauss))
      # lobe_axis = np.zeros((size_pointcloud,self.num_sph_gauss,3))
      # sh_or_sg=False
      # for i in range(self.harmonic_number):
      #   name_r="sh"+str(i)+"r"
      #   name_g="sh"+str(i)+"g"
      #   name_b="sh"+str(i)+"b"
      #   if name_r in fields and name_g in fields and name_b in fields:
      #     sh_or_sg=True
      #     if i==0:
      #       colors[:,0]=pointcloud[name_r]
      #       colors[:,1]=pointcloud[name_g]
      #       colors[:,2]=pointcloud[name_b]
      #     else:
      #       spherical_harmonics[:,:,i-1]=np.vstack((pointcloud[name_r], pointcloud[name_g], pointcloud[name_b])).T
      # for i in range(self.num_sph_gauss):
      #   name_r,name_g,name_b="sg"+str(i)+"r","sg"+str(i)+"g","sg"+str(i)+"b"
      #   name_lx,name_ly,name_lz="lobe_"+str(i)+"x","lobe_"+str(i)+"y","lobe_"+str(i)+"z"
      #   name_bs="bandwidth_sharpness"+str(i)
      #   if name_r in fields and name_g in fields and name_b in fields and name_lx in fields and name_ly in fields and name_lz in fields and name_bs in fields:
      #     sh_or_sg=True
      #     sph_gauss_features[:,:,i]=np.vstack((pointcloud[name_r], pointcloud[name_g], pointcloud[name_b])).T
      #     lobe_axis[:,i,:]=np.vstack((pointcloud[name_lx], pointcloud[name_ly], pointcloud[name_lz])).T
      #     bandwidth_sharpness[:,i]=pointcloud[name_bs]
      # if not(sh_or_sg):
      # print("No color field in the pointcloud")
      # sys.exit(0)
    if 'sx' in fields and 'sy' in fields and 'sz' in fields and not baseline:
      print("Scale field found")
      scales=np.vstack((pointcloud['sx'], pointcloud['sy'], pointcloud['sz'])).T
      self.scales=torch.tensor(scales,device=device,dtype=self.data_type).contiguous().requires_grad_(True)
      self.scales.contiguous().requires_grad_(True)
    else:  
      print("No scale field in the pointcloud")
      dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(points).float().cuda()), 0.0000001)
      self.scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
      self.scales.contiguous().requires_grad_(True)
    if 'qx' in fields and 'qy' in fields and 'qz' in fields and 'qw' in fields and not baseline:
      print("Rotation field found")
      quaternions = np.vstack((pointcloud['qx'], pointcloud['qy'], pointcloud['qz'], pointcloud['qw'])).T
      self.quaternions=torch.tensor(quaternions,device=device,dtype=self.data_type).contiguous().requires_grad_(True)
    else:
      print("No rotation field in the pointcloud")
      quaternions=np.zeros((size_pointcloud,4))
      quaternions[:,0]=1
      self.quaternions=torch.tensor(quaternions,requires_grad=True,device=device,dtype=self.data_type).contiguous()
    
    self.xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)

    filter_3D = np.zeros(points.shape[0])[..., np.newaxis] + 0.0001
    self.filter_3D = torch.log(torch.from_numpy(filter_3D).type(torch.float32).to(device))
  
  def load_from_gaussian_splatting(self,name,num_harmonics):
    print("Loading pointcloud from gaussian splatting file:",name)
    print("num_harmonics : ", num_harmonics)
    pointcloud= ply.read_ply(name)
    # Check if the pointcloud has an x,y,z field
    fields=pointcloud.dtype.names
    if 'x' in fields and 'y' in fields and 'z' in fields:
      print("x,y,z coordinates found")
      points = np.vstack((pointcloud['x'], pointcloud['y'], pointcloud['z'])).T
      size_pointcloud=points.shape[0]
      self.positions=torch.from_numpy(points).type(torch.float32).to(device)
      self.positions=self.positions.contiguous()
      self.positions.requires_grad_(True)
      print("Number of points :", size_pointcloud)
    else:
      print("No x,y,z coordinates in the pointcloud")
      sys.exit(0)
    # Check if the pointcloud has an opacity field
    if 'opacity' in fields:
      self.densities=torch.tensor(pointcloud['opacity'],device=device,dtype=self.data_type).contiguous().requires_grad_(True)

    else:
      print("No opacity field in the pointcloud")
      sys.exit(0)
    if 'scale_0' in fields:
      print("scale_0 field found")
      print("self.quaternions.requires_grad: ",self.quaternions.requires_grad)
      print("self.quaternions.is_leaf: ",self.quaternions.is_leaf)
      self.scales=torch.log(torch.tensor(np.ones((size_pointcloud,3))*0.03,device=device,dtype=self.data_type))
      # self.scales[:,0]=torch.tensor(1.3+pointcloud['scale_0'],device=device,dtype=self.data_type)
      # self.scales[:,1]=torch.tensor(1.3+pointcloud['scale_1'],device=device,dtype=self.data_type)
      # self.scales[:,2]=torch.tensor(1.3+pointcloud['scale_2'],device=device,dtype=self.data_type)
      self.scales[:,0]=torch.tensor(pointcloud['scale_0'],device=device,dtype=self.data_type)
      self.scales[:,1]=torch.tensor(pointcloud['scale_1'],device=device,dtype=self.data_type)
      self.scales[:,2]=torch.tensor(pointcloud['scale_2'],device=device,dtype=self.data_type)
      self.scales=self.scales.contiguous().requires_grad_(True)
    else:
      print("No scale_0 field in the pointcloud")
      sys.exit(0)
      
    if 'rot_0' in fields and 'rot_1' in fields and 'rot_2' in fields and 'rot_3' in fields:
      print("Rotation field found")
      quaternions = np.vstack((pointcloud['rot_0'], pointcloud['rot_1'], pointcloud['rot_2'], pointcloud['rot_3'])).T
      print("quaternions.shape: ",quaternions.shape)
      self.quaternions=torch.tensor(quaternions,device=device,dtype=self.data_type).contiguous().requires_grad_(True)
    else:
      print("No rotation field in the pointcloud")
      self.quaternions=torch.zeros((size_pointcloud,4),dtype=self.data_type,device=device)
      
    # Check if the pointcloud has a color field
    if 'f_dc_0' in fields and 'f_dc_1' in fields and 'f_dc_2' in fields:
      print("Color field found")
      colors = np.vstack((pointcloud['f_dc_0'], pointcloud['f_dc_1'], pointcloud['f_dc_2'])).T
      colors = colors#+0.5*(np.sqrt(np.pi)*2)
      self.rgb=torch.tensor(colors,device=device,dtype=self.data_type).contiguous().requires_grad_(True)
      spherical_harmonics = np.zeros((size_pointcloud,3,num_harmonics-1))
      self.spherical_harmonics=torch.tensor(spherical_harmonics,requires_grad=True,device=device,dtype=self.data_type)
    else:
      print("No color field in the pointcloud")
      sys.exit(0)
    if False: #'f_rest_0' in fields and 'f_rest_1' in fields and 'f_rest_2' and 'f_rest_3' and 'f_rest_4' and 'f_rest_5' and 'f_rest_6' and 'f_rest_7' and 'f_rest_8' in fields and 'f_rest_9' in fields and 'f_rest_10' in fields and 'f_rest_11' in fields and 'f_rest_12' in fields and 'f_rest_13' in fields and 'f_rest_14' in fields and 'f_rest_15' in fields and 'f_rest_16' in fields and 'f_rest_17' in fields and 'f_rest_18' in fields and 'f_rest_19' in fields and 'f_rest_20' in fields and 'f_rest_21' in fields and 'f_rest_22' in fields and 'f_rest_23' in fields:
      print("Rest field found")
      rest_field = np.vstack((pointcloud['f_rest_0'], pointcloud['f_rest_1'], pointcloud['f_rest_2'], pointcloud['f_rest_3'], pointcloud['f_rest_4'], pointcloud['f_rest_5'], pointcloud['f_rest_6'], pointcloud['f_rest_7'], pointcloud['f_rest_8'], pointcloud['f_rest_9'], pointcloud['f_rest_10'], pointcloud['f_rest_11'], pointcloud['f_rest_12'], pointcloud['f_rest_13'], pointcloud['f_rest_14'], pointcloud['f_rest_15'], pointcloud['f_rest_16'], pointcloud['f_rest_17'], pointcloud['f_rest_18'], pointcloud['f_rest_19'], pointcloud['f_rest_20'], pointcloud['f_rest_21'], pointcloud['f_rest_22'], pointcloud['f_rest_23'])).T
      print("rest_field.shape",rest_field.shape)
      # self.spherical_harmonics[:,0,1:]=torch.from_numpy(rest_field[:,0:24:3]).type(torch.float32).to(device)
      # self.spherical_harmonics[:,1,1:]=torch.from_numpy(rest_field[:,1:24:3]).type(torch.float32).to(device)
      # self.spherical_harmonics[:,2,1:]=torch.from_numpy(rest_field[:,2:24:3]).type(torch.float32).to(device)
      self.spherical_harmonics[:,0,1:]=torch.from_numpy(rest_field[:,0:8]).type(torch.float32).to(device)
      self.spherical_harmonics[:,1,1:]=torch.from_numpy(rest_field[:,8:16]).type(torch.float32).to(device)
      self.spherical_harmonics[:,2,1:]=torch.from_numpy(rest_field[:,16:24]).type(torch.float32).to(device)
      self.spherical_harmonics.requires_grad_(True)
      print("self.spherical_harmonics",self.spherical_harmonics)
      # exit(0)

    #Find what is the maximum value of the rest field: f_rest_i
    max_value=0
    for field in fields:
      # Check if field is of the form f_rest_i
      if field[:7]=="f_rest_":
        value_field=int(field[7:])
        max_value=max(max_value,value_field)
    rest_field = np.zeros((size_pointcloud,max_value+1))
    for i in range(max_value+1):
      rest_field[:,i]=pointcloud['f_rest_'+str(i)]
    harmonic_per_channel=(max_value+1)//3
    self.harmonic_number=(max_value+1)//3 + 1
    self.spherical_harmonics=torch.zeros((size_pointcloud,3,self.harmonic_number-1),dtype=torch.float32,device=device)
    self.spherical_harmonics[:,0,:]=torch.from_numpy(rest_field[:,0:harmonic_per_channel]).type(torch.float32).to(device)
    self.spherical_harmonics[:,1,:]=torch.from_numpy(rest_field[:,harmonic_per_channel:2*harmonic_per_channel]).type(torch.float32).to(device)
    self.spherical_harmonics[:,2,:]=torch.from_numpy(rest_field[:,2*harmonic_per_channel:3*harmonic_per_channel]).type(torch.float32).to(device)
    self.spherical_harmonics.contiguous().requires_grad_(True)
    print("self.spherical_harmonics",self.spherical_harmonics)
  
    #else:
    #  print("No rest field in the pointcloud")
    #  sys.exit(0)
      
    self.xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
  
  def set_gaussian(self, position, scale, rotation, color, opacity):
    """for testing numerical stability"""
    position = position.reshape(1,3)
    scale = torch.log(scale.reshape(1,3))
    rotation = rotation.reshape(1,4)
    opacity = opacity.reshape(1,)
      
    self.positions    = nn.Parameter(position.requires_grad_(True))
    self.quaternions  = nn.Parameter(rotation.requires_grad_(True))
    self.scales       = nn.Parameter(scale.requires_grad_(True))
    self.densities    = nn.Parameter(opacity.requires_grad_(True))
    self.rgb          = nn.Parameter(color.reshape(1,3).requires_grad_(True))
    
    self.spherical_harmonics = nn.Parameter(color.reshape(1,3,1).contiguous().requires_grad_(True))
    self.sph_shs_rest        = nn.Parameter(torch.zeros((1,3,2), dtype=torch.float32, device="cuda").contiguous().requires_grad_(True))
    
    self.sph_gauss_features  = nn.Parameter(torch.zeros_like(self.sph_shs_rest).requires_grad_(True))
    self.bandwidth_sharpness = nn.Parameter( torch.zeros((1,1), dtype=self.data_type, device=self.device).requires_grad_(True) )
    self.lobe_axis      = nn.Parameter( (torch.ones((1,3,3), dtype=self.data_type, device=self.device) * torch.eye(3, device=self.device)).requires_grad_(True) )
    
  def prune_points(self,mask):
    "Delete points if they are in the mask"
    mask = ~mask
    
    with torch.no_grad():
      dict_pruned_tensors=self.optim_managers._prune_optimizers(mask)
      self.positions=dict_pruned_tensors["xyz"]
      self.rgb=dict_pruned_tensors["rgb"]
      self.spherical_harmonics=dict_pruned_tensors["sh"]
      self.densities=dict_pruned_tensors["density"]
      self.scales=dict_pruned_tensors["scales"]
      self.quaternions=dict_pruned_tensors["quaternions"]
      self.sph_gauss_features=dict_pruned_tensors["sph_gauss_features"]
      self.bandwidth_sharpness=dict_pruned_tensors["bandwidth_sharpness"]
      self.lobe_axis=dict_pruned_tensors["lobe_axis"]

      self.xyz_gradient_accum_norm=self.xyz_gradient_accum_norm[mask]
      self.num_accum=self.num_accum[mask]
      self.num_accum_gnv=self.num_accum_gnv[mask]
      self.filter_3D=self.filter_3D[mask]

  @torch.no_grad()
  def unlock_spherical_harmonics(self,n_sh):
    """
      Fix n_sh spherical harmonics coefficients to the pointcloud
    """
    print("Unlock spherical harmonics: ",n_sh)
    self.harmonic_number=n_sh
    actual_n_sh=self.spherical_harmonics.shape[2]
    #dict_tensor=self.optim_managers._cat_tensors({"sh":1e-6*torch.randn((len(self.positions),3,n_sh-actual_n_sh-1),dtype=torch.float32,device=device)},dim=2)
    dict_tensor=self.optim_managers._cat_tensors({"sh":0.0*torch.randn((len(self.positions),3,n_sh-actual_n_sh-1),dtype=torch.float32,device=device)},dim=2)
    self.spherical_harmonics=dict_tensor["sh"]

  @torch.no_grad()
  def unlock_spherical_gaussians(self,n_sph_gauss):
    """
      Fix n_sph_gauss spherical gaussians to the pointcloud
    """
    print("Unlock spherical gaussians: ",n_sph_gauss)
    dict_tensor=self.optim_managers._cat_tensors({"sph_gauss_features":torch.zeros((len(self.positions),3,n_sph_gauss),dtype=torch.float32,device=device)},dim=2)
    self.sph_gauss_features=dict_tensor["sph_gauss_features"]
    dict_tensor=self.optim_managers._cat_tensors({"bandwidth_sharpness":torch.zeros((len(self.positions),n_sph_gauss),dtype=torch.float32,device=device)},dim=1)
    self.bandwidth_sharpness=dict_tensor["bandwidth_sharpness"]
    dict_tensor=self.optim_managers._cat_tensors({"lobe_axis":torch.randn((len(self.positions),n_sph_gauss,3),dtype=torch.float32,device=device)},dim=1)
    self.lobe_axis=dict_tensor["lobe_axis"]
    self.num_sph_gauss+=n_sph_gauss

  @torch.no_grad()
  def clamp_density(self):
    self.densities[:] = torch.clamp(self.densities, 0)
  
  @torch.no_grad()
  def select_inside_mask(self,mask):
    xyz=self.positions[mask].contiguous()
    color_features=self.get_color_features().contiguous()
    color_features=color_features[mask].contiguous()
    densities=self.densities[mask].contiguous()
    scales=self.get_scale()
    scales=scales[mask].contiguous()
    quaternions=self.get_normalized_quaternion()
    quaternions=quaternions[mask].contiguous()

    sph_gauss_features=self.sph_gauss_features[mask].contiguous()
    bandwidth_sharpness=self.get_bandwidth_sharpness()
    bandwidth_sharpness=bandwidth_sharpness[mask].contiguous()
    lobe_axis=self.get_normalized_lobe_axis()
    lobe_axis=lobe_axis[mask].contiguous()
    return xyz,color_features,densities,scales,quaternions,sph_gauss_features,bandwidth_sharpness,lobe_axis

    
  def subsample(self, percentage_points):
    with torch.no_grad():
      number_points_before = len(self.positions)
      mask=torch.rand(len(self.positions))<percentage_points
      # dict_pruned_tensors=self.optim_managers._prune_optimizers(mask) // To use later
      #Check if self.quaternions is contiguous
      print("self.quaternions.is_contiguous()",self.quaternions.is_contiguous())
      self.positions=self.positions[mask].requires_grad_(True)
      self.rgb=self.rgb[mask].requires_grad_(True)
      self.spherical_harmonics=self.spherical_harmonics[mask].requires_grad_(True)
      self.densities=self.densities[mask].requires_grad_(True)
      print("self.quaternions.is_contiguous()",self.quaternions.is_contiguous())

      self.scales=self.scales[mask].requires_grad_(True)
      self.quaternions=self.quaternions[mask].requires_grad_(True)
      self.xyz_gradient_accum_norm=self.xyz_gradient_accum_norm[mask]
      self.num_accum=self.num_accum[mask]
      self.num_accum_gnv=self.num_accum_gnv[mask]
      print("Number points deleted (subsample): ",number_points_before-len(self.positions))
      self.sph_gauss_features=self.sph_gauss_features[mask].requires_grad_(True)
      self.bandwidth_sharpness=self.bandwidth_sharpness[mask].requires_grad_(True)
      self.lobe_axis=self.lobe_axis[mask].requires_grad_(True)

  def copyPoints(self,positions,rgb,spherical_harmonics,densities,scales,quaternions,num_sph_gauss,sph_gauss_features,bandwidth_sharpness,lobe_axis):
    """
    Copy the points in the pointcloud with positions, spherical_harmonics and densities attributes
    """
    with torch.no_grad():
      xyz=positions.to(device).requires_grad_()
      rgb=rgb.to(device).requires_grad_()
      spherical_harmonics=spherical_harmonics.to(device).requires_grad_()
      densities=densities.to(device).requires_grad_()
      self.positions=xyz
      self.rgb=rgb
      self.spherical_harmonics=spherical_harmonics
      self.densities=densities
      self.scales=scales.to(device).requires_grad_()
      self.quaternions=quaternions.to(device).requires_grad_()
      self.xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
      self.num_accum=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
      self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)

      self.num_sph_gauss=num_sph_gauss
      self.sph_gauss_features=sph_gauss_features.to(device).requires_grad_()
      self.bandwidth_sharpness=bandwidth_sharpness.to(device).requires_grad_()
      self.lobe_axis=lobe_axis.to(device).requires_grad_()


  def add_points(self,xyz,rgb,sh,sigma,scales,quaternions):
    with torch.no_grad():
      self.positions=torch.cat([self.positions,xyz],dim=0).requires_grad_(True)
      self.rgb=torch.cat([self.rgb,rgb],dim=0).requires_grad_(True)
      self.spherical_harmonics=torch.cat([self.spherical_harmonics,sh],dim=0).requires_grad_(True)
      self.densities=torch.cat([self.densities,sigma],dim=0).requires_grad_(True)
      self.scales = torch.cat([self.scales,scales],dim=0).requires_grad_(True)
      self.quaternions=torch.cat([self.quaternions,quaternions],dim=0).requires_grad_(True)

      self.xyz_gradient_accum_norm=torch.cat([self.xyz_gradient_accum_norm,torch.zeros(len(xyz),dtype=self.data_type,device=device)],dim=0)
      self.num_accum=torch.cat([self.num_accum,torch.zeros(len(xyz),dtype=self.data_type,device=device)],dim=0)
      self.num_accum_gnv=torch.cat([self.num_accum_gnv,torch.zeros(len(xyz),dtype=self.data_type,device=device)],dim=0)

      self.sph_gauss_features=torch.cat([self.sph_gauss_features,torch.zeros((len(xyz),3,self.num_sph_gauss),dtype=self.data_type,device=device)],dim=0).requires_grad_(True)
      self.bandwidth_sharpness=torch.cat([self.bandwidth_sharpness,torch.zeros((len(xyz),self.num_sph_gauss),dtype=self.data_type,device=device)],dim=0).requires_grad_(True)
      self.lobe_axis=torch.cat([self.lobe_axis,torch.zeros((len(xyz),self.num_sph_gauss,3),dtype=self.data_type,device=device)],dim=0).requires_grad_(True)
    
  @torch.no_grad()
  def accumulate_gradient(self,grad):
    self.xyz_gradient_accum_norm += torch.norm(grad,dim=1)
    self.num_accum += torch.where(torch.norm(grad,dim=1)>0,1.0,0.0)
  
  @torch.no_grad()
  def accumulate_gradient_gaussians_not_visible(self,grad):
    self.num_accum_gnv += torch.where(torch.norm(grad,dim=1)>0,1.0,0.0)
      
  @torch.no_grad()
  def reset_gradient_accum(self):
    self.xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum=torch.zeros(len(self.positions),dtype=self.data_type,device=device)

  @torch.no_grad()
  def reset_gradient_accum_gaussians_not_visible(self):
    self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)

  def delete_gaussians_not_seen(self):
    with torch.no_grad():
      keep_points=self.num_accum_gnv>0
      dict_pruned_tensors=self.optim_managers._prune_optimizers(keep_points)
      self.positions=dict_pruned_tensors["xyz"]
      self.rgb=dict_pruned_tensors["rgb"]
      self.spherical_harmonics=dict_pruned_tensors["sh"]
      self.densities=dict_pruned_tensors["density"]
      self.scales=dict_pruned_tensors["scales"]
      self.quaternions=dict_pruned_tensors["quaternions"]
      self.xyz_gradient_accum_norm=self.xyz_gradient_accum_norm[keep_points]
      self.num_accum=self.num_accum[keep_points]
      self.num_accum_gnv=self.num_accum_gnv[keep_points]
      self.filter_3D=self.filter_3D[keep_points]

      self.sph_gauss_features=dict_pruned_tensors["sph_gauss_features"]
      self.bandwidth_sharpness=dict_pruned_tensors["bandwidth_sharpness"]
      self.lobe_axis=dict_pruned_tensors["lobe_axis"]


  def delete_points_too_close_camera(self, mask):
    with torch.no_grad():
      number_points_before = len(self.positions)
      dict_pruned_tensors=self.optim_managers._prune_optimizers(mask)
      self.positions=dict_pruned_tensors["xyz"]
      self.rgb=dict_pruned_tensors["rgb"]
      self.spherical_harmonics=dict_pruned_tensors["sh"]
      self.densities=dict_pruned_tensors["density"]
      self.scales=dict_pruned_tensors["scales"]
      self.quaternions=dict_pruned_tensors["quaternions"]
      self.xyz_gradient_accum_norm=self.xyz_gradient_accum_norm[mask]
      self.num_accum=self.num_accum[mask]
      self.num_accum_gnv=self.num_accum_gnv[mask]

      self.sph_gauss_features=dict_pruned_tensors["sph_gauss_features"]
      self.bandwidth_sharpness=dict_pruned_tensors["bandwidth_sharpness"]
      self.lobe_axis=dict_pruned_tensors["lobe_axis"]
      print("Number points deleted (too close from cameras): ",number_points_before-len(self.positions))

  def get_color_features(self):
    return torch.cat([self.rgb[:,:,None],self.spherical_harmonics],dim=2)
  
  def get_scale(self):
    scales = torch.exp(self.scales)
    return scales

  def get_inv_scale(self):
    return torch.exp(-self.scales)
  
  def get_normalized_quaternion(self):
    return self.quaternions/torch.norm(self.quaternions,dim=1)[:,None]
  
  @torch.no_grad()
  def normalize_quaternion(self):
    norm=torch.norm(self.quaternions,dim=1)
    self.quaternions[norm==0]= torch.tensor([1.0,0.0,0.0,0.0],dtype=self.data_type,device=device,requires_grad=True)
    self.quaternions/=torch.norm(self.quaternions,dim=1)[:,None]
  
  def get_density(self):
    # return torch.nn.functional.softplus(self.densities)
    return self.densities
  
  def get_normalized_lobe_axis(self):
    #lobe axis is unnormalized vector
    norm=torch.norm(self.lobe_axis,dim=2)
    with torch.no_grad():
      self.lobe_axis[norm==0]= torch.tensor([0.0,0.0,1.0],dtype=self.data_type,device=device,requires_grad=True)
    unit_lobe_axis= self.lobe_axis/torch.norm(self.lobe_axis,dim=2)[:,:,None]
    return unit_lobe_axis
  
  @torch.no_grad()
  def normalize_lobe_axis(self):
    norm=torch.norm(self.lobe_axis,dim=2)
    self.lobe_axis[norm==0]= torch.tensor([0.0,0.0,1.0],dtype=self.data_type,device=device,requires_grad=True)
    self.lobe_axis/=torch.norm(self.lobe_axis,dim=2)[:,:,None]

  def get_bandwidth_sharpness(self):
    return torch.nn.functional.softplus(self.bandwidth_sharpness)
  
  def get_data(self,normalized_quaternion=True,normalized_lobe_axis=True):
    if normalized_quaternion:
      quaternions=self.get_normalized_quaternion()
    else:
      quaternions=self.quaternions
    if normalized_lobe_axis:
      lobe_axis=self.get_normalized_lobe_axis()
    else:
      lobe_axis=self.lobe_axis
    return (self.positions, self.get_scale(), quaternions, self.get_density(),self.get_color_features(),
      self.sph_gauss_features, self.get_bandwidth_sharpness(), lobe_axis)
  
  def get_data_inv_scale(self,normalized_quaternion=True,normalized_lobe_axis=True):
    if normalized_quaternion:
      quaternions=self.get_normalized_quaternion()
    else:
      quaternions=self.quaternions
    if normalized_lobe_axis:
      lobe_axis=self.get_normalized_lobe_axis()
    else:
      lobe_axis=self.lobe_axis
    return (self.positions, self.get_inv_scale(), quaternions, self.get_density(),self.get_color_features(),
      self.sph_gauss_features, self.get_bandwidth_sharpness(), lobe_axis)

  def init_random_debug(self):
    num_pts=2
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    xyz[0,:]=np.array([0.0,0.0,0.0])
    xyz[1,:]=xyz[0,:]+np.array([0.0,0.0,0.1])
    fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
    shs = np.random.random((num_pts, 3)) / 255.0
    shs[0,:]=np.array([1.0,0.0,0.0])
    shs[1,:]=np.array([0.0,1.0,0.0])
    fused_color = (torch.tensor(np.asarray(shs)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, self.harmonic_number)).float().cuda()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0

    scales=torch.ones((fused_point_cloud.shape[0],3),dtype=torch.float,device="cuda")
    scales[:,0]*=torch.log(torch.tensor(0.11*2))
    scales[:,1]*=torch.log(torch.tensor(0.11*2))
    scales[:,2]*=torch.log(torch.tensor(0.35*2))
    opacities = inverse_sigmoid(0.99 * torch.ones((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda"))
    # quaternions=torch.rand((fused_point_cloud.shape[0],4),dtype=self.data_type,device=device)
    quaternions=torch.zeros((fused_point_cloud.shape[0],4),dtype=self.data_type,device="cuda")
    quaternions[:,0]=1
    quaternions[1,0]=1/np.sqrt(2)
    quaternions[1,1]=1/np.sqrt(2)
    self.scales=nn.Parameter(scales.cuda().contiguous().requires_grad_(True))
    self.quaternions=nn.Parameter(quaternions.contiguous().requires_grad_(True))
    self.positions=nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
    self.densities=nn.Parameter(opacities.contiguous().requires_grad_(True))
    self.rgb=nn.Parameter(features[:,:,0].contiguous().requires_grad_(True))
    self.spherical_harmonics=nn.Parameter(features[:,:,1:].contiguous().requires_grad_(True))
    self.xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)


  def init_random(self,num_pts,sh_nb,sg_nb,default_density,spatial_lr_scale=0):
    self.harmonic_number=sh_nb
    self.num_sph_gauss=sg_nb
    #Bounding box for NeRF Synthetic scene:
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    shs = np.random.random((num_pts, 3)) / 255.0

    # rgb=SH2RGB(torch.tensor(shs).float().cuda())
    # self.spatial_lr_scale = spatial_lr_scale
    fused_point_cloud = torch.tensor(np.asarray(xyz)).float().cuda()
    # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    fused_color = (torch.tensor(np.asarray(shs)).float().cuda())
    
    features = torch.zeros((fused_color.shape[0], 3, self.harmonic_number)).float().cuda()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

    # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda"))
    densities=torch.ones((fused_point_cloud.shape[0]),dtype=torch.float,device="cuda")*default_density

    self.positions=nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
    self.densities=nn.Parameter(densities.contiguous().requires_grad_(True))
    self.rgb=nn.Parameter(features[:,:,0].contiguous().requires_grad_(True))
    self.spherical_harmonics=nn.Parameter(features[:,:,1:].contiguous().requires_grad_(True))
    # self.quaternions=nn.Parameter(radius.requires_grad_(True))
    self.scales=nn.Parameter(scales.cuda().contiguous().requires_grad_(True))
    quaternions=torch.zeros((fused_point_cloud.shape[0],4),dtype=self.data_type,device=device)
    quaternions[:,0]=1

    self.quaternions=nn.Parameter(quaternions.contiguous().requires_grad_(True))
    
    self.xyz_gradient_accum_norm=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.filter_3D = torch.log(torch.ones(fused_point_cloud.shape[0],1).cuda())
    
    self.sph_gauss_features=torch.zeros((len(self.positions),3,self.num_sph_gauss),dtype=self.data_type,device=device,requires_grad=True)
    self.bandwidth_sharpness=torch.zeros((len(self.positions),self.num_sph_gauss),dtype=self.data_type,device=device,requires_grad=True)
    self.lobe_axis=torch.zeros((len(self.positions),self.num_sph_gauss,3),dtype=self.data_type,device=device,requires_grad=True)
    
  def check_nan_grad(self):
        # Check if there are nan values
    if torch.sum(torch.isnan(self.densities.grad))>0:
      print("There are nan values in the gradients of the densities")
      exit(0)
    if torch.sum(torch.isnan(self.positions.grad))>0:
      print("There are nan values in the gradients of the positions")
      exit(0)
    # if torch.sum(torch.isnan(self.quaternions.grad))>0:
      # print("There are nan values in the gradients of the quaternions")
      # exit(0)
    if torch.sum(torch.isnan(self.scales.grad))>0:
      print("There are nan values in the gradients of the scales")
      exit(0)
    if torch.sum(torch.isnan(self.quaternions.grad))>0:
      print("There are nan values in the gradients of the quaternions")
      exit(0)
    if torch.sum(torch.isnan(self.rgb.grad))>0:
      print("There are nan values in the gradients of the rgb")
      exit(0)
    if torch.sum(torch.isnan(self.spherical_harmonics.grad))>0:
      print("There are nan values in the gradients of the spherical harmonics")
      exit(0)
    if torch.sum(torch.isnan(self.sph_gauss_features.grad))>0:
      print("There are nan values in the gradients of the sph_gauss_features")
      exit(0)
    if torch.sum(torch.isnan(self.bandwidth_sharpness.grad))>0:
      print("There are nan values in the gradients of the bandwidth_sharpness")
      exit(0)
    if torch.sum(torch.isnan(self.lobe_axis.grad))>0:
      print("There are nan values in the gradients of the lobe_axis")
      exit(0)

  def check_required_grad(self):
    if not self.positions.requires_grad:
      print("self.positions.requires_grad: ",self.positions.requires_grad)
      exit(0)
    if not self.rgb.requires_grad:
      print("self.rgb.requires_grad: ",self.rgb.requires_grad)
      exit(0)
    if not self.spherical_harmonics.requires_grad:
      print("self.spherical_harmonics.requires_grad: ",self.spherical_harmonics.requires_grad)
      exit(0)
    if not self.densities.requires_grad:
      print("self.densities.requires_grad: ",self.densities.requires_grad)
      exit(0)
    if not self.scales.requires_grad:
      print("self.scales.requires_grad: ",self.scales.requires_grad)
      exit(0)
    if not self.quaternions.requires_grad:
      print("self.quaternions.requires_grad: ",self.quaternions.requires_grad)
      exit(0)
    if not self.sph_gauss_features.requires_grad:
      print("self.sph_gauss_features.requires_grad: ",self.sph_gauss_features.requires_grad)
      exit(0)
    if not self.bandwidth_sharpness.requires_grad:
      print("self.bandwidth_sharpness.requires_grad: ",self.bandwidth_sharpness.requires_grad)
      exit(0)
    if not self.lobe_axis.requires_grad:
      print("self.lobe_axis.requires_grad: ",self.lobe_axis.requires_grad)
      exit(0)
      
  def assign_grad(self,grad_dict):
    with torch.no_grad():
      self.densities.grad=grad_dict["density"]
      self.positions.grad=grad_dict["xyz"]
      self.scales.grad=grad_dict["scales"]
      self.quaternions.grad=grad_dict["quaternions"]
      self.rgb.grad=grad_dict["rgb"]
      self.spherical_harmonics.grad=grad_dict["sh"]
      self.sph_gauss_features.grad=grad_dict["sph_gauss_features"]
      self.bandwidth_sharpness.grad=grad_dict["bandwidth_sharpness"]
      self.lobe_axis.grad=grad_dict["lobe_axis"]
      
  
  def print_shape(self):
    print("self.positions.shape: ",self.positions.shape)
    print("self.rgb.shape: ",self.rgb.shape)
    print("self.spherical_harmonics.shape: ",self.spherical_harmonics.shape)
    print("self.densities.shape: ",self.densities.shape)
    print("self.quaternions.shape: ",self.quaternions.shape)
    print("self.scales.shape: ",self.scales.shape)
    print("self.quaternions.shape: ",self.quaternions.shape)
    print("self.xyz_gradient_accum_norm.shape: ",self.xyz_gradient_accum_norm.shape)
    print("self.num_accum.shape: ",self.num_accum.shape)
    print("self.sph_gauss_features.shape: ",self.sph_gauss_features.shape)
    print("self.bandwidth_sharpness.shape: ",self.bandwidth_sharpness.shape)
    print("self.lobe_axis.shape: ",self.lobe_axis.shape)
  
  def save_model(self,iteration,save_folder):
    print("Saving model at iteration ",iteration)
    data=(self.harmonic_number,
    self.positions,
    self.rgb,
    self.spherical_harmonics,
    self.densities,
    self.scales,
    self.quaternions,
    self.sph_gauss_features,
    self.bandwidth_sharpness,
    self.lobe_axis,
    self.num_sph_gauss,
    self.xyz_gradient_accum_norm,
    self.num_accum,
    self.optim_managers.get_state_dict(),
    iteration)
    torch.save(data, save_folder + "/chkpnt" + str(iteration) + ".pth")
  
  def restore_model(self,iteration,checkpoint_folder,config_opti=None):
    print("Restoring model at iteration ",iteration)
    data=torch.load(checkpoint_folder + "/chkpnt" + str(iteration) + ".pth",weights_only=False)
    self.harmonic_number=data[0]
    self.positions=data[1]
    self.rgb=data[2]
    self.spherical_harmonics=data[3]
    self.densities=data[4]
    self.scales=data[5]
    self.quaternions=data[6]
    self.sph_gauss_features=data[7]
    self.bandwidth_sharpness=data[8]
    self.lobe_axis=data[9]
    self.num_sph_gauss=data[10]
    self.xyz_gradient_accum_norm=data[11]
    self.num_accum=data[12]
    self.num_accum_gnv=torch.zeros(len(self.positions),dtype=self.data_type,device=device)
    self.filter_3D=torch.log(torch.ones(len(self.positions),1).cuda())
    if config_opti is not None:
      # [optim_manag_positions,optim_manag_rgb,optim_manag_spherical_harmonics,optim_manag_densities,optim_manag_quaternions,optim_manag_scales]=define_optimizer_manager(config_opti,self,[],[],[],[],[],[])
      # self.setup_optimizers({"xyz":optim_manag_positions,"rgb":optim_manag_rgb,"sh":optim_manag_spherical_harmonics,"density":optim_manag_densities,"scales":optim_manag_scales,"quaternions":optim_manag_quaternions})
      [optim_manag_positions,optim_manag_rgb,optim_manag_spherical_harmonics,optim_manag_densities,optim_manag_scales,optim_manag_quaternions,optim_manag_sph_gauss_features,optim_manag_bandwidth_sharpness,optim_manag_lobe_axis]=define_optimizer_manager(config_opti,self,[],[],[],[],[],[],[],[],[])
      self.setup_optimizers({"xyz":optim_manag_positions,"rgb":optim_manag_rgb,"sh":optim_manag_spherical_harmonics,"density":optim_manag_densities,
                                        "scales":optim_manag_scales,"quaternions":optim_manag_quaternions,
                                        "sph_gauss_features":optim_manag_sph_gauss_features,
                                        "bandwidth_sharpness":optim_manag_bandwidth_sharpness,
                                        "lobe_axis":optim_manag_lobe_axis})
      self.optim_managers.load_state_dict(data[13])
    return data[14]
  
  def project_points(self,camera):
    """ camera.world_view_transform: (4,4) matrix such that:
    camera.world_view_transform[:3,:3] is the rotation matrix from camera to world coordinates but using it as x^T*R is equivalent to R^T*x 
     torch.matmul(self.positions, camera.world_view_transform[:3,:3]) transform a vector from world to camera coordinates
     torch.matmul(points, camera.world_view_transform[:3,:3]) + camera.world_view_transform[3,:3] transform a point from world to camera coordinates"""
    pos_ref_cam = torch.matmul(self.positions, camera.world_view_transform[:3,:3]) + camera.world_view_transform[3,:3]

    return pos_ref_cam

  def smallest_discernible_distance(self,camera):
    pos_ref_cam = self.project_points(camera)
    lx,ly=2*math.tan(camera.FoVx / 2)/camera.image_width,2*math.tan(camera.FoVy / 2)/camera.image_height
    z_candidate=pos_ref_cam[:,2].clone()
    z_candidate=z_candidate[z_candidate>camera.znear]
    Lx,Ly=lx*z_candidate,ly*z_candidate
    return Lx.min().item(),Ly.min().item()
  
  def rescale_points(self,scaling_factor):
    with torch.no_grad():
      self.positions *= scaling_factor
      self.scales += torch.log(torch.tensor(scaling_factor))
  
  def check_contiguous(self):
    try :
      assert self.positions.is_contiguous()
      assert self.rgb.is_contiguous()
      assert self.spherical_harmonics.is_contiguous()
      assert self.densities.is_contiguous()
      assert self.scales.is_contiguous()
      assert self.quaternions.is_contiguous()
      assert self.xyz_gradient_accum_norm.is_contiguous()
      assert self.num_accum.is_contiguous()
      assert self.sph_gauss_features.is_contiguous()
      assert self.bandwidth_sharpness.is_contiguous()
      assert self.lobe_axis.is_contiguous()
    except:
      print("Not contiguous")
      print("Which:")
      if not self.positions.is_contiguous():
        print("self.positions")
      if not self.rgb.is_contiguous():
        print("self.rgb")
      if not self.spherical_harmonics.is_contiguous():
        print("self.spherical_harmonics")
      if not self.densities.is_contiguous():
        print("self.densities")
      if not self.scales.is_contiguous():
        print("self.scales")
      if not self.quaternions.is_contiguous():
        print("self.quaternions")
      if not self.xyz_gradient_accum_norm.is_contiguous():
        print("self.xyz_gradient_accum_norm")
      if not self.num_accum.is_contiguous():
        print("self.num_accum")
      if not self.sph_gauss_features.is_contiguous():
        print("self.sph_gauss_features")
      if not self.bandwidth_sharpness.is_contiguous():
        print("self.bandwidth_sharpness")
      if not self.lobe_axis.is_contiguous():
        print("self.lobe_axis")
      exit(0)

  @torch.no_grad()
  def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                            new_sph_gauss_features, new_bandwidth_sharpness, new_lobe_axis,new_filter_3D, quiet=True):
      d = {"xyz": new_xyz,
      "rgb": new_features_dc,
      "sh": new_features_rest,
      "density": new_opacities,
      "scales" : new_scaling,
      "quaternions" : new_rotation,
      "sph_gauss_features": new_sph_gauss_features,
      "bandwidth_sharpness": new_bandwidth_sharpness,
      "lobe_axis": new_lobe_axis}
      
      dict_tensor=self.optim_managers._cat_tensors(d,dim=0)

      self.positions = dict_tensor["xyz"]
      self.rgb = dict_tensor["rgb"]
      self.spherical_harmonics = dict_tensor["sh"]
      self.densities = dict_tensor["density"]
      self.scales = dict_tensor["scales"]
      self.quaternions = dict_tensor["quaternions"]
      self.sph_gauss_features = dict_tensor["sph_gauss_features"]
      self.bandwidth_sharpness = dict_tensor["bandwidth_sharpness"]
      self.lobe_axis = dict_tensor["lobe_axis"]

      self.xyz_gradient_accum_norm = torch.zeros((self.positions.shape[0]), device="cuda")
      self.num_accum = torch.zeros((self.positions.shape[0]), device="cuda")
      new_num_accum_gnv = torch.zeros((len(new_xyz)), device="cuda")
      self.num_accum_gnv = torch.cat((self.num_accum_gnv,new_num_accum_gnv),dim=0)
      self.filter_3D = torch.cat((self.filter_3D,new_filter_3D),dim=0)
      #Nombre de points ajoutés
      if not quiet:
        print("Number of points added: ",len(new_xyz))

  @torch.no_grad()
  def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, quiet=True):
      if not quiet:
        print("Densify and split")
      n_init_points = self.positions.shape[0]
      # Extract points that satisfy the gradient condition
      padded_grad = torch.zeros((n_init_points), device="cuda")
      padded_grad[:grads.shape[0]] = grads.squeeze()
      selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

      selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scale(), dim=1).values > 0.01*scene_extent)

      stds=self.get_scale()[selected_pts_mask].repeat(N,1)
      means =torch.zeros((stds.size(0), 3),device="cuda")
      samples = torch.normal(mean=means, std=stds)
      rots = build_rotation(self.quaternions[selected_pts_mask]).repeat(N,1,1)
      
      new_filter_3D = self.filter_3D[selected_pts_mask].repeat(N, 1)
      new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.positions[selected_pts_mask].repeat(N, 1)
      new_scaling = torch.log(self.get_scale()[selected_pts_mask].repeat(N,1) / (0.8*N))
      new_rotation = self.quaternions[selected_pts_mask].repeat(N,1)
      new_features_dc = self.rgb[selected_pts_mask].repeat(N,1)
      new_features_rest = self.spherical_harmonics[selected_pts_mask].repeat(N,1,1)
      new_opacity = self.densities[selected_pts_mask].repeat(N)/2
      new_sph_gauss_features = self.sph_gauss_features[selected_pts_mask].repeat(N,1,1)
      new_bandwidth_sharpness = self.bandwidth_sharpness[selected_pts_mask].repeat(N,1)
      new_lobe_axis = self.lobe_axis[selected_pts_mask].repeat(N,1,1)

      self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, 
                                  new_sph_gauss_features, new_bandwidth_sharpness, new_lobe_axis, new_filter_3D, quiet)
    
      prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
      self.prune_points(prune_filter)

  @torch.no_grad()   
  def densify_and_clone(self, grads, grad_threshold, scene_extent, quiet=True):
      if not quiet:
        print("Densify and clone")
      # 3D Gaussian splatting code
      # Extract points that satisfy the gradient condition
      # selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
      selected_pts_mask = torch.where(grads >= grad_threshold, True, False)
      selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.get_scale(), dim=1).values <= 0.01*scene_extent)
      
      self.densities[selected_pts_mask] = self.densities[selected_pts_mask]/2
      
      new_filter_3D = self.filter_3D[selected_pts_mask]
      new_xyz = self.positions[selected_pts_mask]
      new_features_dc = self.rgb[selected_pts_mask]
      new_features_rest = self.spherical_harmonics[selected_pts_mask]
      new_opacities = self.densities[selected_pts_mask]
      new_scaling = self.scales[selected_pts_mask]
      new_rotation = self.quaternions[selected_pts_mask]

      new_sph_gauss_features = self.sph_gauss_features[selected_pts_mask]
      new_bandwidth_sharpness = self.bandwidth_sharpness[selected_pts_mask]
      new_lobe_axis = self.lobe_axis[selected_pts_mask]
      self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
                                new_sph_gauss_features, new_bandwidth_sharpness, new_lobe_axis, new_filter_3D, quiet)

  @torch.no_grad()
  def densify_and_prune(self, max_grad, min_density, extent, quiet=True):
    grads = self.xyz_gradient_accum_norm / self.num_accum
    grads[grads.isnan()] = 0.0
    if not quiet:
      print("mean grads : ", torch.mean(grads))

    self.densify_and_clone(grads, max_grad, extent, quiet=quiet)
    self.densify_and_split(grads, max_grad, extent, quiet=quiet)

    prune_mask = (self.densities <= min_density).squeeze()

    self.prune_points(prune_mask)
    if not quiet:
      print("Number of points pruned: ",torch.sum(prune_mask).item())
      print("Number of points remaining: ",len(self.positions))
    torch.cuda.empty_cache()
    
  def reset_density(self):
    densities_new = ((torch.min(self.densities.detach() , torch.ones_like(self.densities.detach())*4.0)).contiguous())
    self.densities = densities_new.requires_grad_(True)
    dicts_replace={"density":self.densities}
    self.optim_managers._replace(dicts_replace)


  @torch.no_grad()
  def compute_3D_filter(self, cameras):
    #print("Computing 3D filter")
    #consider focal length and image width
    xyz = self.positions
    distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
    valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
    
    # we should use the focal length of the highest resolution camera
    focal_length = 0.
    for camera in cameras:
      # transform points to camera space
      R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
      T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
        # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
      xyz_cam = xyz @ R + T[None, :]
      
      #xyz_to_cam = torch.norm(xyz_cam, dim=1)
      
      # project to screen space
      valid_depth = xyz_cam[:, 2] > 0.2
      
      
      x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
      z = torch.clamp(z, min=0.001)
      
      focal_x = fov2focal(camera.FoVx,camera.image_width)
      focal_y = fov2focal(camera.FoVy,camera.image_height)
      x = x / z * focal_x + camera.image_width / 2.0
      y = y / z * focal_y + camera.image_height / 2.0
      
      # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
      
      # use similar tangent space filtering as in the paper
      in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
  
      valid = torch.logical_and(valid_depth, in_screen)
      
      # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
      distance[valid] = torch.min(distance[valid], z[valid])
      valid_points = torch.logical_or(valid_points, valid)
      if focal_length < focal_x:
          focal_length = focal_x
      if focal_length < focal_y:
          focal_length = focal_y

    # print("distance.shape: ",distance.shape)
    # print("valid_points.shape: ",valid_points.shape)
    
    distance[~valid_points] = distance[valid_points].max()
    
    #TODO remove hard coded value
    #TODO box to gaussian transform
    filter_3D = 0.5 * distance / focal_length #* (0.2 ** 0.5)
    self.filter_3D = torch.log(filter_3D[..., None])
    # exp_scales = torch.exp(self.scales)
    # min_scales = filter_3D[..., None].flatten()
    # with torch.no_grad():
    #   exp_scales[:,0] = torch.max(exp_scales[:,0],min_scales)
    #   exp_scales[:,1] = torch.max(exp_scales[:,1],min_scales)
    #   exp_scales[:,2] = torch.max(exp_scales[:,2],min_scales)
    #   self.scales = torch.log(exp_scales)
    #   self.scales.contiguous().requires_grad_(True)
    return
