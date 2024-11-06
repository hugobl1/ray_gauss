import numpy as np
import torch
import os
from scripts.ply import write_ply
import argparse
from torch.profiler import *

import classes.point_cloud as point_cloud

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='Convert pt tensor to py file.')
parser.add_argument('-output', required=True, help="Path to output folder")
parser.add_argument('-iter', required=True,help='Checkpoint number')
parser.add_argument('-save_folder',default="saved_pc",help="Path to save the file")
args = parser.parse_args()

#Create the folder to save the ply file
save_folder=os.path.join(args.output,args.save_folder)
os.makedirs(save_folder,exist_ok=True)

model_path=os.path.join(args.output,"models","chkpnt"+str(args.iter)+".pth")
data=torch.load(model_path,weights_only=False)
# data=torch.load(args.output + "/model" + str(args.iter) + ".pth")

# restore_model
harmonic_number=data[0]
pc=data[1].detach().cpu().numpy()
rgb=data[2].detach().cpu().numpy()
spherical_harmonics=data[3].detach().cpu().numpy()
colors=np.concatenate((rgb[:,:,None],spherical_harmonics),axis=2)
densities=data[4].detach().cpu().numpy()
scales=data[5].detach().cpu().numpy()
quaternions=data[6].detach().cpu().numpy()
sph_gauss_features=data[7].detach().cpu().numpy()
bandwidth_sharpness=data[8].detach().cpu().numpy()
lobe_axis=data[9].detach().cpu().numpy()
num_sph_gauss=data[10]
xyz_gradient_accum_norm=data[11].detach().cpu().numpy()
num_accum=data[12].detach().cpu().numpy()

#Print the shape of each tensor
# print("harmonic_number: ",harmonic_number)
# print("self.harmonic_number: ",data[0])
# print("self.positions.shape: ",data[1].shape)
# print("self.rgb.shape: ",data[2].shape)
# print("self.spherical_harmonics.shape: ",data[3].shape)
# print("self.densities.shape: ",data[4].shape)
# print("self.scales.shape: ",data[5].shape)
# print("self.quaternions.shape: ",data[6].shape)
# print("self.sph_gauss_features.shape: ",data[7].shape)
# print("self.bandwidth_sharpness.shape: ",data[8].shape)
# print("self.lobe_axis.shape: ",data[9].shape)
# print("num_sph_gauss: ",data[10])
# print("self.xyz_gradient_accum_norm.shape: ",data[11].shape)
# print("self.num_accum: ",data[12].shape)


size_pc=pc.shape[0]
number_channels=3

print("Number of points: ",size_pc)
fields=['x','y','z']
data=[]
data.append(pc)



for channel in range(number_channels):
    for harmonic in range(harmonic_number):
        data.append(colors[:,channel,harmonic])
        if channel==0:
            fields.append("sh"+str(harmonic)+"r")
        elif channel==1:
            fields.append("sh"+str(harmonic)+"g")
        elif channel==2:
            fields.append("sh"+str(harmonic)+"b")
fields.append("density")
data.append(densities)

fields.extend(['sx','sy','sz'])
data.append(scales)

fields.extend(['qx','qy','qz','qw'])
data.append(quaternions)

for channel in range(number_channels):
    for sph_gauss in range(num_sph_gauss):
        data.append(sph_gauss_features[:,channel,sph_gauss])
        if channel==0:
            fields.append("sg"+str(sph_gauss)+"r")
        elif channel==1:
            fields.append("sg"+str(sph_gauss)+"g")
        elif channel==2:
            fields.append("sg"+str(sph_gauss)+"b")

for sph_gauss in range(num_sph_gauss):
    fields.append("bandwidth_sharpness"+str(sph_gauss)) 
    data.append(bandwidth_sharpness[:,sph_gauss])

for sph_gauss in range(num_sph_gauss):
    # fields.append("lobe_x","lobe_y","lobe_z")
    # fields.extend(["lobe_x"+str(sph_gauss),"lobe_y"+str(sph_gauss),"lobe_z"+str(sph_gauss)])
    # fields.append("lobe_"+str(sph_gauss)+"x","lobe_"+str(sph_gauss)+"y","lobe_"+str(sph_gauss)+"z")
    fields.extend(["lobe_"+str(sph_gauss)+"x","lobe_"+str(sph_gauss)+"y","lobe_"+str(sph_gauss)+"z"])
    data.append(lobe_axis[:,sph_gauss])

fields.append("gradient_accum_norm")
data.append(xyz_gradient_accum_norm)

# name_save=args.save_folder+"/"+"point_cloud_iter"+str(args.iter)+".ply"
name_save=os.path.join(save_folder,"point_cloud_iter"+str(args.iter)+".ply")
print("name_save: ",name_save)
write_ply(name_save,data,fields)

