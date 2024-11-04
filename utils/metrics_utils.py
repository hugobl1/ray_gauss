import torch 
from math import log10, sqrt
import numpy as np

def PSNR(original, compressed):
    #if tensor use torch.mean else use np.mean
    if torch.is_tensor(original):
        mse= torch.mean((original - compressed) ** 2)
    else:
        mse = np.mean((original - compressed) ** 2)
    # mse = torch.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr