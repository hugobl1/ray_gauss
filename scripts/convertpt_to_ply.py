from threadpoolctl import _ALL_INTERNAL_APIS
import numpy as np
import torch

from scripts.ply import write_ply
import argparse, os
from torch.profiler import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='Convert pt tensor to py file.')
parser.add_argument('-folder', default="saved_tensors", help='folder')
parser.add_argument('-pc', default="positions", help='Positions')
parser.add_argument('-col', default="spherical_harmonics", help='Colors')
parser.add_argument('-op', default="densities", help='Opacities')
parser.add_argument('-scale', default="scales", help='scale')
parser.add_argument('-iter', default="", help='Iter')
parser.add_argument('-save',default="pointcloud",help="Path to save the file")
parser.add_argument('-save_folder',default="saved_pc",help="Path to save the file")
args = parser.parse_args()

#Check that args.folder exists
if not os.path.exists(args.folder):
    print("Folder not found")
    exit()

name_op=args.folder+"/"+args.op+".pt"
name_pc=args.folder+"/"+args.pc+".pt"
name_col=args.folder+"/"+args.col+".pt"
name_scale=args.folder+"/"+args.scale+".pt"
# name_gradient_accum=args.folder+"/"+name_pc+"_xyz_gradient_accum_norm.pt"

name_save=args.folder+"/"+args.save+".ply"
if args.iter!="":
    print("Be careful, iter argument will overwrite other arguments")
    iter=str(args.iter)
    name_op=args.folder+"/"+args.op+"_iter"+iter+".pt"
    name_pc=args.folder+"/"+args.pc+"_iter"+iter+".pt"
    name_col=args.folder+"/"+args.col+"_iter"+iter+".pt"
    name_scale=args.folder+"/"+args.scale+"_iter"+iter+".pt"
    name_gradient_accum=args.folder+"/"+args.pc+"_iter"+iter+"_xyz_gradient_accum_norm"+".pt"
    name_save=args.save_folder+"/"+"point_cloud_iter"+iter+".ply"
    


opacities=torch.load(name_op, map_location=torch.device('cpu'),weights_only=False).detach().numpy()

pc=torch.load(name_pc, map_location=torch.device('cpu'),weights_only=False).detach().numpy()
                                                #colors[:pc.shape[0],:,0]/(2*np.sqrt(np.pi))

gradient_accum=torch.load(name_gradient_accum, map_location=torch.device('cpu'),weights_only=False).detach().numpy()

if args.col!="none":
    colors=torch.load(name_col, map_location=torch.device('cpu'),weights_only=False).detach().numpy()
else:
    print("No colors")
    colors=np.zeros((pc.shape[0],3,1))

scales=torch.load(name_scale, map_location=torch.device('cpu'),weights_only=False).detach().numpy()

use_rgb=False

size_pc=pc.shape[0]
number_channels=colors.shape[1]
number_harmonics=colors.shape[2]
print("Number of points: ",size_pc)
fields=['x','y','z']
data=[]
data.append(pc)

fields.append("gradient_accum_norm")
data.append(gradient_accum[:size_pc])

fields.append("scale_x")
data.append(scales[:size_pc,0])
fields.append("scale_y")
data.append(scales[:size_pc,1])
fields.append("scale_z")
data.append(scales[:size_pc,2])

for channel in range(number_channels):
    for harmonic in range(number_harmonics):
        data.append(colors[:size_pc,channel,harmonic]*1/(2*np.sqrt(np.pi)))
        if channel==0:
            fields.append("sh"+str(harmonic)+"r")
        elif channel==1:
            fields.append("sh"+str(harmonic)+"g")
        elif channel==2:
            fields.append("sh"+str(harmonic)+"b")
fields.append("opacity")
data.append(opacities[:size_pc])

# Add rgb channels by multiplying colors[:size_pc,:,0] by 1/2sqrt(pi)

if(use_rgb):
    data.append(colors[:size_pc,0,0]/(2*np.sqrt(np.pi)))
    data.append(colors[:size_pc,1,0]/(2*np.sqrt(np.pi)))
    data.append(colors[:size_pc,2,0]/(2*np.sqrt(np.pi)))
    fields.append("red")
    fields.append("green")
    fields.append("blue")
# print("Fields: ",fields)
# print("Data: ",data)
write_ply(name_save,data,fields)

