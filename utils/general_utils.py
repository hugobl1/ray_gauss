
import os
import imageio
import numpy as np
import json
import torch
import math
import matplotlib.pyplot as plt
#from asyncio.windows_events import NULL
from turtle import st
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from scipy import ndimage
import scripts.ply as ply


import cupy as cp
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

device = "cuda" if torch.cuda.is_available() else "cpu"



def get_data(root="../nerf_example_data/nerf_synthetic/lego", stage="train",
             white_background=False,dilatation=True,number_dilatations=2):
  """ get_data récupère les paramètres essentielles du dataset traité.

  Args:
    root: string donnant le chemin pointant vers le dataset traité
    stage: string indiquant quel ensemble de données doit être traité
           Valeurs prévues:{"train","val","test"}
    background: booléen indiquant si l'on utilisera un backgroud
      spécifique(à retravailler valable seulement pour False)
    dilatation: booléen indiquant si l'on dilate le filtre binaire
      issue du masque de transparencence
    number_dilatations: int indiquant le nombre d'itérations de
      dilatation réalisé

  Returns:
    focal: float donnant la focale de la caméra
    all_camera_to_world: array de shape (N,4,4) donnant les matrices
      de transformation de la caméra vers le monde,N étant le nombre de caméras
    all_groundtruth: array de shape (N,H,W,3) donnant les images de références,
      N étant le nombre de caméras, H et W les dimensions des images
    all_alpha_mask: array de shape (N,H,W) donnant les masques de
      transparence, N étant le nombre de caméras, H et W les dimensions des images
    all_alpha_mask_01: array de shape (N,H,W) donnant les filtres
     issues des masques de transparence, N étant le nombre de caméras,
     H et W les dimensions des images
    all_alpha_mask_border01: array de shape (N,H,W) donnant
      les filtres de la zone de dilatation seulement issues des masques
      de transparence, N étant le nombre de caméras, H et W les dimensions des images
  """
  all_camera_to_world = []
  all_groundtruth = []
  all_alpha_mask=[]
  all_alpha_mask01=[]
  all_alpha_mask_border01=[]
  images_path = os.path.join(root, stage)
  transforms_path = os.path.join(root, "transforms_" + stage + ".json")
  with open(transforms_path, "r") as f:
    transforms = json.load(f)
  #j = json.load(open(transforms_path, "r"))
  for frame in transforms["frames"]:
    fpath = os.path.join(images_path, os.path.basename(frame["file_path"]) + ".png")
    #fpath = os.path.join(images_path, os.path.basename(frame["file_path"]))
    camera_to_world = frame["transform_matrix"]
    image_groundtruth = imageio.imread(fpath).astype(np.float32) / 255.0
    #double its size
    # image_groundtruth=np.repeat(np.repeat(image_groundtruth,2,axis=0),2,axis=1)
    alpha_mask=np.copy(image_groundtruth[..., -1])
    alpha_mask01=np.copy(image_groundtruth[..., -1]>0.1)
    if dilatation:
      alpha_mask_border01=ndimage.binary_dilation(alpha_mask01,iterations=number_dilatations)
      # alpha_mask_border01=np.logical_xor(alpha_mask_border01,alpha_mask01)
    if white_background:
      image_groundtruth = image_groundtruth[...,:3]*image_groundtruth[...,-1:] + (1.-image_groundtruth[...,-1:])
    else:
      image_groundtruth = image_groundtruth[..., :3] * image_groundtruth[..., 3:]
    all_camera_to_world.append(camera_to_world)
    all_groundtruth.append(image_groundtruth)
    all_alpha_mask.append(alpha_mask)
    all_alpha_mask01.append(alpha_mask01)
    if dilatation:
      all_alpha_mask_border01.append(alpha_mask_border01)
  focal = 0.5 * all_groundtruth[0].shape[1] / np.tan(0.5 * transforms["camera_angle_x"])
  all_camera_to_world = np.asarray(all_camera_to_world)
  all_groundtruth = np.asarray(all_groundtruth)
  all_alpha_mask=np.asarray(all_alpha_mask)
  all_alpha_mask01=np.asarray(all_alpha_mask01)
  if dilatation:
    all_alpha_mask_border01=np.asarray(all_alpha_mask_border01)
  return focal, all_camera_to_world, all_groundtruth,all_alpha_mask,all_alpha_mask01,all_alpha_mask_border01


def get_rays_np(height, width, focal, c2w):
  i, j = np.meshgrid((np.arange(width, dtype=np.float32) + 0.5),
                     (np.arange(height, dtype=np.float32) + 0.5), indexing="xy")  
  dirs = np.stack([(i-width*.5)/focal, -(j-height*.5)/focal, -np.ones_like(i)], -1)
  # Rotate ray directions from camera frame to the world frame
  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
  rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
  # Translate camera frame's origin to the world frame. It is the origin of all rays.
  rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
  return rays_o, rays_d

def get_rays_torch(height, width, focal, camera_to_world):
  """Returns ray origins and directions in the world frame,
     each of shape (H, W, 3) for a pinhole camera model.

  Args:
    height: int, image height
    width: int, image width
    focal: float, focal length
    camera_to_world: (4, 4) array, camera-to-world matrix

  Returns:
    rays_origin_tensor: (H, W, 3) tensor, ray origins in the world frame
    rays_direction_tensor: (H, W, 3) tensor, ray directions in the world frame
  """
  camera_to_world_tensor=torch.from_numpy(camera_to_world)
  it,jt=torch.meshgrid((torch.arange(width, dtype=torch.float32) + 0.5),
                       (torch.arange(height, dtype=torch.float32) + 0.5),
                        indexing="xy")
  directions_tensor=torch.stack([(it-width*.5)/focal, -(jt-height*.5)/focal, -torch.ones_like(it)], -1)
  # Rotate ray directions from camera frame to the world frame
  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
  rays_direction_tensor = torch.sum(directions_tensor[..., None, :] * camera_to_world_tensor[:3, :3], -1)
  # Translate camera frame's origin to the world frame. It is the origin of all rays.
  rays_origin_tensor=torch.broadcast_to(camera_to_world_tensor[:3, -1], rays_direction_tensor.shape)
  return rays_origin_tensor, rays_direction_tensor

def displacement_in_pixel(directions,height,width,focal,camera_to_world):
  """Return the displaced ray directions in the world frame
  Args:
    directions: (N,3) tensor, ray directions in the world frame
    height: int, image height
    width: int, image width
    focal: float, focal length
    camera_to_world: (N,4,4) array, camera-to-world matrix
  
  """
  if (len(camera_to_world.shape)==2):
    camera_to_world=camera_to_world[None,:,:]
  if (len(directions.shape)==1):
    directions=directions[None,:]
  dxdy=torch.tensor([0.5/focal,0.5/focal,0])
  dxdy=torch.rand(directions.shape[0],3)*dxdy[None,:]
  displaced_directions=directions+torch.bmm(camera_to_world[:,:3,:3],dxdy[:,:,None]).squeeze(-1)
  return displaced_directions

def get_cameras_centers(rays_or_dir):
  len_rays = len(rays_or_dir)
  centers = np.zeros((len_rays, 3))
  for i in range(len_rays):
  # all cameras share same center
    centers[i, :] = rays_or_dir[i][0][0, 0]
  return centers



def reshape_numpy(arr, red_fac):
  height, width = arr.shape[:2]
  im = Image.fromarray(np.uint8(255*arr))
  im = im.resize((height//red_fac, width//red_fac), Image.ANTIALIAS)
  return np.array(im)/255.0

def reshape_torch(image, reduction_factor):
  """ reshape and convert a numpy array of shape (H,W,3) to a torch tensor of
      shape (H//reduction_factor,W//reduction_factor,3)

  Args:
    image (numpy array): image to reshape and convert to torch tensor
    reduction_factor (int): reduction factor for the images

  Returns:
    image_torch: torch tensor of shape
      (H//reduction_factor,W//reduction_factor,3)
  """
  height,width = image.shape[:2]
  image = Image.fromarray(np.uint8(255*image))
  image = image.resize((height//reduction_factor, width//reduction_factor),
                       Image.ANTIALIAS)
  image_torch=torch.from_numpy(np.array(image)/255.0)
  return image_torch

def get_images_origins_directions(all_camera_to_world, all_groundtruth,
                                  focal, reduction_factor=1):
  """ get images origins and directions for all images in the dataset
      in the world system of coordinates

  Args:
    all_camera_to_world: numpy array de shape (N,4,4) donnant les matrices
     de transformation de la caméra vers le monde, N étant le nombre de caméras
    all_groundtruth: numpy array de shape (N,H,W,3) donnant les images de références,
      N étant le nombre de caméras, H et W les dimensions des images
    focal (float): focal length
    reduction_factor (int): reduction factor for the images

  Returns:
    reduced_images: numpy array ou tenseur PyTorch de shape
      (N,H//reduction_factor,W//reduction_factor,3) donnant les images
      de références réduites, N étant le nombre de caméras, H et W les dimensions des images
    origin_direction_rays: si to_torch==True tenseur PyTorch de shape (N,2,H,W,3) donnant
      les origines et directions des rayons, tel que:
      -origin_direction_rays[i,0,j,k,:] donne l'origine du rayon de coordonnées (j,k) de l'image i
      -origin_direction_rays[i,1,j,k,:] donne la direction du rayon de coordonnées (j,k) de l'image i
  """
  height, width = all_groundtruth[0].shape[:2]
  to_numpy,to_torch=False,True
  if to_numpy:
    origin_direction_rays = []
    reduced_images = [reshape_numpy(gt, reduction_factor) for gt in all_groundtruth]
    for c2w in all_camera_to_world:
      ray_np = get_rays_np(height, width, focal, c2w)
      oris = ray_np[0][::reduction_factor, ::reduction_factor]
      direct = ray_np[1][::reduction_factor, ::reduction_factor]
      origin_direction_rays.append((oris, direct))
  if to_torch:
    origin_direction_rays=torch.zeros(len(all_groundtruth),2,height//reduction_factor,width//reduction_factor,3)
    # origin_direction_rays=torch.zeros(len(all_groundtruth),2,2*height//reduction_factor,2*width//reduction_factor,3)
    reduced_images=torch.zeros(len(all_groundtruth),height//reduction_factor,width//reduction_factor,3)
    for l in range(len(all_groundtruth)):
      reduced_images[l] = reshape_torch(all_groundtruth[l], reduction_factor)

      ray_torch = get_rays_torch(height, width, focal, all_camera_to_world[l])
      origin = ray_torch[0][::reduction_factor, ::reduction_factor]
      direction = ray_torch[1][::reduction_factor, ::reduction_factor]
      origin_direction_rays[l,0]=origin
      origin_direction_rays[l,1]= direction
  return reduced_images, origin_direction_rays


def regular_3d_indexes(n):
  i = np.arange(n)
  j = np.arange(n)
  k = np.arange(n)
  return np.transpose([np.tile(i, len(j)*len(k)), np.tile(np.repeat(j, len(i)), len(k)), np.repeat(k, len(i)*len(j))])


def rolling_average(p, k=100):
  p2 = np.zeros((p.shape[0]-k))
  for i in range(k):
    p2 += p[i:(p.shape[0]-k+i)]
  return p2/k


def compute_psnr(grid, disp_rays_test, disp_ims_test, number_points=500):
  m = np.zeros(len(disp_ims_test))
  for i in tqdm(range(len(disp_ims_test))):
    with torch.no_grad():
      new_im = grid.render_large_image_from_rays(
          disp_rays_test[i], (number_points, 1.2))
      m[i] = peak_signal_noise_ratio(
          new_im, disp_ims_test[i].astype("float32"))
  return m.mean()


def normalize01(t, m=0, M=1):
  t = torch.minimum(t, torch.tensor(M, device=device))
  t = torch.maximum(t, torch.tensor(m, device=device))
  return t

def patchify(tensor, patch_size):
  """ patchify a tensor

  Args:
    tensor (torch tensor): tensor of shape (N,H,W,C) or (N,H,W) to patchify
    patch_size (int): size of the patch

  Returns:
    tensor_patchified (torch tensor): tensor of shape (N*H*W//patch_size**2,patch_size,patch_size,C) or (N*H*W//patch_size**2,patch_size,patch_size)
  """
  if(len(tensor.shape)==3):
    tensor_patchified = tensor.unfold(1, patch_size, patch_size).unfold(
      2, patch_size, patch_size).reshape(-1, patch_size, patch_size)
  elif(len(tensor.shape)==4):
    tensor_patchified = tensor.unfold(1, patch_size, patch_size).unfold(
      2, patch_size, patch_size).reshape(-1, 3, patch_size, patch_size).permute(0, 2, 3, 1)
  else:
    raise Exception("tensor must be of shape (N,H,W,C) or (N,H,W)")
  return tensor_patchified

# DATASETS

class IndexDataset(Dataset):
  """Dataset for indices
  """
  def __init__(self, index):
    """Inits the dataset."""
    self.index = index
    
  def __getitem__(self, index):
    index_extracted = self.index[index]
    return index_extracted

  def __len__(self):
    return len(self.index)

class RayDataset(Dataset):
  """Dataset for the rays
  """
  def __init__(self, target_images, origin_direction_rays, all_alphamask, all_alphamask01,all_camera_to_world,focal):
    """Inits the dataset."""
    self.image_height, self.image_width = target_images[0].shape[:2]
    self.number_images=len(target_images)
    self.focal=torch.tensor(focal,dtype=torch.float32)

    rays_origin=origin_direction_rays[:,0].reshape((len(origin_direction_rays)*self.image_height *self.image_width ,3))
    rays_direction=origin_direction_rays[:,1].reshape((len(origin_direction_rays)*self.image_height *self.image_width ,3))
    images=target_images.reshape((len(origin_direction_rays)*self.image_height *self.image_width ,3))
    alphamasks_tensor=torch.from_numpy(all_alphamask)
    alphamasks_tensor=alphamasks_tensor.reshape((len(origin_direction_rays)*self.image_height *self.image_width))

    self.tensor_rays_direction=rays_direction
    self.tensor_rays_origin=rays_origin
    self.tensor_target_pixels=images
    self.alphavalue=alphamasks_tensor
    self.dirnorm=torch.linalg.norm(self.tensor_rays_direction,dim=1,keepdim=True)
    
  def __getitem__(self, index):
    rays_direction=self.tensor_rays_direction[index]
    rays_origin=self.tensor_rays_origin[index]
    return rays_origin,rays_direction, self.tensor_target_pixels[index],self.alphavalue[index]

  def __len__(self):
    return len(self.tensor_rays_direction)
  
  def to(self,device):
    self.tensor_rays_direction=self.tensor_rays_direction.to(device)
    self.tensor_rays_origin=self.tensor_rays_origin.to(device)
    self.tensor_target_pixels=self.tensor_target_pixels.to(device)
    self.alphavalue=self.alphavalue.to(device)
    self.dirnorm=self.dirnorm.to(device)
    return self

  def normalize_ray_directions(self):
    """ Normalize the ray direction """
    self.tensor_rays_direction=self.tensor_rays_direction/self.dirnorm
    print("Ray direction normalized")


def cumsum_by_group(x,idx):
  classic_cumsum=torch.cumsum(x,0)
  print("cumsum",classic_cumsum)

  # Find where idx[i-1] != idx[i]
  idx_diff=torch.where(idx[1:]-idx[:-1]!=0)[0]
  print(idx_diff)
  classic_cumsum_idx_diff=classic_cumsum[idx_diff]

  print("classic_cumsum_idx_diff",classic_cumsum_idx_diff)

  # Count the number of occurences of each idx
  idx_count=torch.bincount(idx)
  print(idx_count)
  classic_cumsum_idx_diff=torch.repeat_interleave(classic_cumsum_idx_diff,idx_count[1:],0)
  print("test",classic_cumsum_idx_diff)
  classic_cumsum_idx_diff=(torch.cat((torch.zeros(idx_count[0]),classic_cumsum_idx_diff)))
  cumsum_by_group=classic_cumsum-classic_cumsum_idx_diff
  return cumsum_by_group

def compute_transmittance(x,idx):
  """
  Input:
  x: (N) tensor of values
  idx: (N) tensor of indices

  Returns:
      cumsum_by_group: (N) tensor of values with cumsum_by_group[i] = sum_{j=a}^{i-1} x[j] where a is the first index of the same group as idx[i]

  """
  classic_cumsum=torch.cumsum(x,0)

  # Find where idx[i-1] != idx[i]
  idx_diff=torch.where(idx[1:]-idx[:-1]!=0)[0]

  classic_cumsum_idx_diff=classic_cumsum[idx_diff]
  # Count the number of occurences of each idx
  idx_count=torch.bincount(idx)

  classic_cumsum_idx_diff=torch.repeat_interleave(classic_cumsum_idx_diff,idx_count[1:])

  classic_cumsum_idx_diff=(torch.cat((torch.zeros((idx_count[0]),device=device),classic_cumsum_idx_diff)))
  
  cumsum_shifted=torch.cumsum(torch.cat((torch.zeros((1),device=device),x[:-1])),0)
  cumsum_by_group=cumsum_shifted-classic_cumsum_idx_diff
  return cumsum_by_group

def compute_transmittance_grad(x,idx):
  """
  Input:
  x: (N) tensor of values
  idx: (N) tensor of indices

  Returns:
  result: (N) tensor of values with result[i] = sum_{j=i+1}^{Nk} x[j] where Nk is the last index of the same group as idx[i]
  """
  #classic_cumsum=torch.cumsum(x,0)
  
  x0=x[:,0]
  x1=x[:,1]
  x2=x[:,2]
  
  classic_cumsum0=torch.cumsum(x0,0)
  classic_cumsum1=torch.cumsum(x1,0)
  classic_cumsum2=torch.cumsum(x2,0)
  classic_cumsum=torch.stack((classic_cumsum0,classic_cumsum1,classic_cumsum2),1)

  # Find where idx[i-1] != idx[i]
  idx_diff=torch.where(idx[1:]-idx[:-1]!=0)[0]

  classic_cumsum_idx_diff=classic_cumsum[idx_diff]
  # Count the number of occurences of each idx
  idx_count=torch.bincount(idx)
  classic_cumsum_idx_diff=torch.repeat_interleave(classic_cumsum_idx_diff,idx_count[1:],0)

  classic_cumsum_idx_diff=(torch.cat((torch.zeros((idx_count[0],3),device=device),classic_cumsum_idx_diff)))
  
  cumsum_by_group=classic_cumsum-classic_cumsum_idx_diff
  max_by_group=cumsum_by_group[torch.cat((idx_diff,torch.tensor([-1],device=device)))]
  
  repeat_max_by_group=torch.repeat_interleave(max_by_group,idx_count,0)
  

  result=repeat_max_by_group-cumsum_by_group
  return result

def save_ply_rgb(points,colors,densities,name):
    # Transfer to cpu and numpy if needed and detach if needed
    if type(points) is torch.Tensor:
      points=points.detach().cpu().numpy()
    if type(colors) is torch.Tensor:
      colors=colors.detach().cpu().numpy()
    if type(densities) is torch.Tensor:
      densities=densities.detach().cpu().numpy()
    # Save the pointcloud
    ply.write_ply(name, [points,colors,densities], ['x', 'y', 'z','red','green','blue','density'])


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
      
#Define cupy2torch that converts an unknown number of cupy arrays to torch tensors
def cupy2torch(*args):
    # return [from_dlpack(x.toDlpack()) for x in args]
    return[torch.from_dlpack(x) for x in args]
  
#Define torch2cupy that converts an unknown number of torch tensors to cupy arrays
def torch2cupy(*args):
    # return [cp.fromDlpack(to_dlpack(x)) for x in args]
    return [cp.from_dlpack(x.detach()) for x in args]

def inverse_sigmoid(x):
    return torch.log(x/(1-x))
  
  
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def reduce_supersampling(target_width,tartget_height,ray_colors,factor):
  """ Reduce the supersampled image by a factor k
  """
  factor_x,factor_y=factor
  #Sum each factor_x values
  ray_colors_sum=ray_colors[0::factor_x]
  for i in range(1,factor_x):
    ray_colors_sum+=ray_colors[i::factor_x]
  ray_colors_sum=ray_colors_sum.reshape((tartget_height*factor_y,target_width,3))
  #Sum each factor_y values
  ray_colors_mean=ray_colors_sum[0::factor_y]
  for i in range(1,factor_y):
    ray_colors_mean+=ray_colors_sum[i::factor_y]
  return ray_colors_mean/(factor_x*factor_y)
  # ray_colors_sum=ray_colors[0::2]+ray_colors[1::2]
  # width=viewpoint_cam.image_width
  # height=viewpoint_cam.image_height
  # ray_colors_sum=ray_colors_sum.reshape((2*height,width,3))
  # ray_colors_mean=(ray_colors_sum[0::2]+ray_colors_sum[1::2])/4