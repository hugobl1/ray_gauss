#  Copyright 2021 PlenOctree Authors.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.

    :param deg: int SH max degree. Currently, 0-4 supported
    :param sh: torch.Tensor SH coeffs (..., C, (max degree + 1) ** 2)
    :param dirs: torch.Tensor unit directions (..., 3)

    :return: (..., C)
    """
    assert deg <= 4 and deg >= 0
    assert (deg + 1) ** 2 == sh.shape[-1]
    C = sh.shape[-2]

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * xy * z * sh[..., 10] +
                        C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def eval_sh_bases(deg, dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be 
    obtained through simple multiplication.

    :param deg: int SH max degree. Currently, 0-4 supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., (deg+1) ** 2)
    """
    assert deg <= 4 and deg >= 0
    result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y
        result[..., 2] = C1 * z
        result[..., 3] = -C1 * x
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy
            result[..., 5] = C2[1] * yz
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = C2[3] * xz
            result[..., 8] = C2[4] * (xx - yy)

            if deg > 2:
                result[..., 9] = C3[0] * y * (3 * xx - yy)
                result[..., 10] = C3[1] * xy * z
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = C3[5] * z * (xx - yy)
                result[..., 15] = C3[6] * x * (xx - 3 * yy)

                if deg > 3:
                    result[..., 16] = C4[0] * xy * (xx - yy)
                    result[..., 17] = C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5


def backward_sh(pos,campos, sh, dL_dRGB):
    """
    Compute the gradient of SH coefficients w.r.t. RGB color.

    :param pos: torch.Tensor (..., 3) position of the point
    :param campos: torch.Tensor (..., 3) camera position
    :param sh: torch.Tensor (..., (deg+1) ** 2) SH coefficients
    :param dL_dRGB: torch.Tensor (..., 3) gradient of loss w.r.t. RGB color

    :return: torch.Tensor (..., (deg+1) ** 2) gradient of loss w.r.t. SH coefficients
    """
    dir_orig = pos - campos
    dirs = dir_orig / torch.norm(dir_orig, dim=-1, keepdim=True)
    deg = int((sh.shape[-1]) ** 0.5) - 1
    assert deg <= 4 and deg >= 0
    C = sh.shape[-2]

    dL_dsh = torch.zeros_like(sh)
    # dRGBdx = torch.zeros_like(sh)
    # dRGBdy = torch.zeros_like(sh)
    # dRGBdz = torch.zeros_like(sh)
    dRGBdx = torch.zeros((sh.shape[0],3),device=sh.device)
    dRGBdy = torch.zeros((sh.shape[0],3),device=sh.device)
    dRGBdz = torch.zeros((sh.shape[0],3),device=sh.device)
    x, y, z = dirs.unbind(-1)
    dRGBdsh0 = C0
    dL_dsh[..., 0] = dRGBdsh0 * dL_dRGB
    if deg > 0:
        dRGBdsh1 = -C1 * y
        dRGBdsh2 = C1 * z
        dRGBdsh3 = -C1 * x

        dL_dsh[..., 1] = dRGBdsh1[:,None] * dL_dRGB
        dL_dsh[..., 2] = dRGBdsh2[:,None] * dL_dRGB
        dL_dsh[..., 3] = dRGBdsh3[:,None] * dL_dRGB

        dRGBdx = -C1 * sh[..., 3]
        dRGBdy = -C1 * sh[..., 1]
        dRGBdz = C1 * sh[..., 2]

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z

            dRGBdsh4 = C2[0] * xy
            dRGBdsh5 = C2[1] * yz
            dRGBdsh6 = C2[2] * (2.0 * zz - xx - yy)
            dRGBdsh7 = C2[3] * xz
            dRGBdsh8 = C2[4] * (xx - yy)
            dL_dsh[..., 4] = dRGBdsh4[:,None] * dL_dRGB
            dL_dsh[..., 5] = dRGBdsh5[:,None] * dL_dRGB
            dL_dsh[..., 6] = dRGBdsh6[:,None] * dL_dRGB
            dL_dsh[..., 7] = dRGBdsh7[:,None] * dL_dRGB
            dL_dsh[..., 8] = dRGBdsh8[:,None] * dL_dRGB
            
            dRGBdx += C2[0] * y[:,None] * sh[..., 4] + C2[2] * 2.0 * -x[:,None] * sh[..., 6] + C2[3] * z[:,None] * sh[..., 7] + C2[4] * 2.0 * x[:,None] * sh[..., 8]
            dRGBdy += C2[0] * x[:,None] * sh[..., 4] + C2[1] * z[:,None] * sh[..., 5] + C2[2] * 2.0 * -y[:,None] * sh[..., 6] + C2[4] * 2.0 * -y[:,None] * sh[..., 8]
            dRGBdz += C2[1] * y[:,None] * sh[..., 5] + C2[2] * 2.0 * 2.0 * z[:,None] * sh[..., 6] + C2[3] * x[:,None] * sh[..., 7]
            # print('dL_dsh',dL_dsh.shape)
            # print('dL_dRGB',dL_dRGB.shape)
            # print("dL_sh",dL_dsh[..., 1].shape)
            # print("dRGBdsh1",dRGBdsh1.shape)
            # print("dL_dRGB",dL_dRGB.shape)
            # print("dRGBdx",dRGBdx.shape)
            # print("dRGBdy",dRGBdy.shape)
            # print("dRGBdz",dRGBdz.shape)
            # print("x",x.shape)
            # print("y",y.shape)
            # print("z",z.shape)
            # exit(0)
            if deg > 2:
                dRGBdsh9 = C3[0] * y * (3.0 * xx - yy)
                dRGBdsh10 = C3[1] * xy * z
                dRGBdsh11 = C3[2] * y * (4.0 * zz - xx - yy)
                dRGBdsh12 = C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                dRGBdsh13 = C3[4] * x * (4.0 * zz - xx - yy)
                dRGBdsh14 = C3[5] * z * (xx - yy)
                dRGBdsh15 = C3[6] * x * (xx - 3.0 * yy)
                dL_dsh[..., 9] = dRGBdsh9[:,None] * dL_dRGB
                dL_dsh[..., 10] = dRGBdsh10[:,None] * dL_dRGB
                dL_dsh[..., 11] = dRGBdsh11[:,None] * dL_dRGB
                dL_dsh[..., 12] = dRGBdsh12[:,None] * dL_dRGB
                dL_dsh[..., 13] = dRGBdsh13[:,None] * dL_dRGB
                dL_dsh[..., 14] = dRGBdsh14[:,None] * dL_dRGB
                dL_dsh[..., 15] = dRGBdsh15[:,None] * dL_dRGB

                dRGBdx += (
                    C3[0] * sh[..., 9] * 3.0 * 2.0 * xy[:,None] +
                    C3[1] * sh[..., 10] * yz[:,None] +
                    C3[2] * sh[..., 11] * -2.0 * xy[:,None] +
                    C3[3] * sh[..., 12] * -3.0 * 2.0 * xz[:,None] +
                    C3[4] * sh[..., 13] * (-3.0 * xx[:,None] + 4.0 * zz[:,None] - yy[:,None]) +
                    C3[5] * sh[..., 14] * 2.0 * xz[:,None] +
                    C3[6] * sh[..., 15] * 3.0 * (xx[:,None] - yy[:,None]))

                dRGBdy += (
                    C3[0] * sh[..., 9] * 3.0 * (xx[:,None] - yy[:,None]) +
                    C3[1] * sh[..., 10] * xz[:,None] +
                    C3[2] * sh[..., 11] * (-3.0 * yy[:,None] + 4.0 * zz[:,None] - xx[:,None]) +
                    C3[3] * sh[..., 12] * -3.0 * 2.0 * yz[:,None] +
                    C3[4] * sh[..., 13] * -2.0 * xy[:,None] +
                    C3[5] * sh[..., 14] * -2.0 * yz[:,None] +
                    C3[6] * sh[..., 15] * -3.0 * 2.0 * xy[:,None])
                
                dRGBdz += (
                    C3[1] * sh[..., 10] * xy[:,None] +
                    C3[2] * sh[..., 11] * 4.0 * 2.0 * yz[:,None] +
                    C3[3] * sh[..., 12] * 3.0 * (2.0 * zz[:,None] - xx[:,None] - yy[:,None]) +
                    C3[4] * sh[..., 13] * 4.0 * 2.0 * xz[:,None] +
                    C3[5] * sh[..., 14] * (xx[:,None] - yy[:,None]))
    #Take care of direction normalization
    #dL_dRGB is the gradient of loss w.r.t. RGB color of shape (..., 3)
    #dRGBdx, dRGBdy, dRGBdz are the gradients of RGB color w.r.t. SH coefficients of shape (..., 3, (deg+1) ** 2)
    #dL_ddir is the gradient of loss w.r.t. direction of shape (..., 3)
    # dL_ddir = torch.stack([torch.sum(dRGBdx * dL_dRGB[..., None], dim=-2), torch.sum(dRGBdy * dL_dRGB[..., None], dim=-2), torch.sum(dRGBdz * dL_dRGB[..., None], dim=-2)], dim=-1)
    
    # print(dRGBdx.shape)
    # print(dL_dRGB.shape)
    
    # print(dL_ddir.shape)
    # dRGB_ddir=torch.stack([dRGBdx.sum(-1),dRGBdy.sum(-1),dRGBdz.sum(-1)],dim=-1)
    # print(dRGB_ddir.shape)
    # print(dL_dRGB.shape)
    # print("sh",sh.shape)
    # print(dRGBdx.shape)
    dL_ddir = torch.stack([(dL_dRGB*dRGBdx).sum(-1), (dL_dRGB* dRGBdy).sum(-1), (dL_dRGB*dRGBdz).sum(-1)], dim=-1)
#     __forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
# {
# 	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
# 	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

# 	float3 dnormvdv;
# 	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
# 	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
# 	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
# 	return dnormvdv;
# }
    sum2 = torch.sum(dir_orig ** 2, dim=-1)
    invsum32 = 1.0 / torch.sqrt(sum2 * sum2 * sum2)
    dL_dmean = torch.stack([((sum2 - dir_orig[..., 0] ** 2) * dL_ddir[..., 0] - dir_orig[..., 1] * dir_orig[..., 0] * dL_ddir[..., 1] - dir_orig[..., 2] * dir_orig[..., 0] * dL_ddir[..., 2]) * invsum32,
                            (-dir_orig[..., 0] * dir_orig[..., 1] * dL_ddir[..., 0] + (sum2 - dir_orig[..., 1] ** 2) * dL_ddir[..., 1] - dir_orig[..., 2] * dir_orig[..., 1] * dL_ddir[..., 2]) * invsum32,
                            (-dir_orig[..., 0] * dir_orig[..., 2] * dL_ddir[..., 0] - dir_orig[..., 1] * dir_orig[..., 2] * dL_ddir[..., 1] + (sum2 - dir_orig[..., 2] ** 2) * dL_ddir[..., 2]) * invsum32], dim=-1)
    # dL_dmean = dnormvdv
    # print("dL_dmean",dL_dmean.shape)
    return dL_dsh, dL_dmean