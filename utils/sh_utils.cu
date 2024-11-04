// #include "vec_math.h"
// #include "helpers.h"

//Declare the constants 

__device__ __constant__ float C0 = 0.28209479177387814;
__device__ __constant__ float C1 = 0.4886025119029199;
__device__ __constant__ float C2[5] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};
__device__ __constant__ float C3[7] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

extern "C" __global__ void forward_sh(float* campos,float3 *positions, float* sh, 
    float* sph_gauss_features, float* bandwidth_sharpness, float* lobe_axis, 
    unsigned int num_sph_gauss,int sh_degree,
    int num_points,
    float* color_features){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    float3 dir_orig = positions[idx] - make_float3(campos[0],campos[1],campos[2]);
    float3 dirs = dir_orig / length(dir_orig);

    float dir_x=dirs.x;
    float dir_y=dirs.y;
    float dir_z=dirs.z;

    float result_red = 0.0f;
    float result_green = 0.0f;
    float result_blue = 0.0f;

    for(int l=0;l<num_sph_gauss;l++){
        float sg_x=sph_gauss_features[idx*num_sph_gauss*3+l];
        float sg_y=sph_gauss_features[idx*num_sph_gauss*3+l+num_sph_gauss];
        float sg_z=sph_gauss_features[idx*num_sph_gauss*3+l+2*num_sph_gauss];
        float sharpness = bandwidth_sharpness[idx*num_sph_gauss+l];
        float3 axis=make_float3(lobe_axis[idx*num_sph_gauss*3+l*3],lobe_axis[idx*num_sph_gauss*3+l*3+1],lobe_axis[idx*num_sph_gauss*3+l*3+2]);
        float dot_product_axis = dot(dirs,axis);
        float gaussian=exp(sharpness*(dot_product_axis-1.0f));
        result_red+=sg_x*gaussian;
        result_green+=sg_y*gaussian;
        result_blue+=sg_z*gaussian;
    }

    int num_sh= (sh_degree+1)*(sh_degree+1);
    int idx_red=idx*num_sh*3;
    int idx_green=idx*num_sh*3+num_sh;
    int idx_blue=idx*num_sh*3+2*num_sh;
    result_red+=sh[idx_red]*C0;
    result_green+=sh[idx_green]*C0;
    result_blue+=sh[idx_blue]*C0;
    if(sh_degree>0){
        result_red=result_red-C1*dir_y*sh[idx_red+1]+C1*dir_z*sh[idx_red+2]-C1*dir_x*sh[idx_red+3];
        result_green=result_green-C1*dir_y*sh[idx_green+1]+C1*dir_z*sh[idx_green+2]-C1*dir_x*sh[idx_green+3];
        result_blue=result_blue-C1*dir_y*sh[idx_blue+1]+C1*dir_z*sh[idx_blue+2]-C1*dir_x*sh[idx_blue+3];
        if(sh_degree>1){
            float xx=dir_x*dir_x,yy=dir_y*dir_y,zz=dir_z*dir_z;
            float xy=dir_x*dir_y,yz=dir_y*dir_z,xz=dir_x*dir_z;
            result_red=result_red+
                C2[0]*xy*sh[idx_red+4]+
                C2[1]*yz*sh[idx_red+5]+
                C2[2]*(2.0*zz-xx-yy)*sh[idx_red+6]+
                C2[3]*xz*sh[idx_red+7]+
                C2[4]*(xx-yy)*sh[idx_red+8];
            result_green=result_green+
                C2[0]*xy*sh[idx_green+4]+
                C2[1]*yz*sh[idx_green+5]+
                C2[2]*(2.0*zz-xx-yy)*sh[idx_green+6]+
                C2[3]*xz*sh[idx_green+7]+
                C2[4]*(xx-yy)*sh[idx_green+8];
            result_blue=result_blue+
                C2[0]*xy*sh[idx_blue+4]+
                C2[1]*yz*sh[idx_blue+5]+
                C2[2]*(2.0*zz-xx-yy)*sh[idx_blue+6]+
                C2[3]*xz*sh[idx_blue+7]+
                C2[4]*(xx-yy)*sh[idx_blue+8];
            if(sh_degree>2){
                result_red=result_red+
                    C3[0]*dir_y*(3.0*xx-yy)*sh[idx_red+9]+
                    C3[1]*xy*dir_z*sh[idx_red+10]+
                    C3[2]*dir_y*(4.0*zz-xx-yy)*sh[idx_red+11]+
                    C3[3]*dir_z*(2.0*zz-3.0*xx-3.0*yy)*sh[idx_red+12]+
                    C3[4]*dir_x*(4.0*zz-xx-yy)*sh[idx_red+13]+
                    C3[5]*dir_z*(xx-yy)*sh[idx_red+14]+
                    C3[6]*dir_x*(xx-3.0*yy)*sh[idx_red+15];
                result_green=result_green+
                    C3[0]*dir_y*(3.0*xx-yy)*sh[idx_green+9]+
                    C3[1]*xy*dir_z*sh[idx_green+10]+
                    C3[2]*dir_y*(4.0*zz-xx-yy)*sh[idx_green+11]+
                    C3[3]*dir_z*(2.0*zz-3.0*xx-3.0*yy)*sh[idx_green+12]+
                    C3[4]*dir_x*(4.0*zz-xx-yy)*sh[idx_green+13]+
                    C3[5]*dir_z*(xx-yy)*sh[idx_green+14]+
                    C3[6]*dir_x*(xx-3.0*yy)*sh[idx_green+15];
                result_blue=result_blue+
                    C3[0]*dir_y*(3.0*xx-yy)*sh[idx_blue+9]+
                    C3[1]*xy*dir_z*sh[idx_blue+10]+
                    C3[2]*dir_y*(4.0*zz-xx-yy)*sh[idx_blue+11]+
                    C3[3]*dir_z*(2.0*zz-3.0*xx-3.0*yy)*sh[idx_blue+12]+
                    C3[4]*dir_x*(4.0*zz-xx-yy)*sh[idx_blue+13]+
                    C3[5]*dir_z*(xx-yy)*sh[idx_blue+14]+
                    C3[6]*dir_x*(xx-3.0*yy)*sh[idx_blue+15];
            }
        }
    }
    color_features[idx*3]=result_red+0.5;
    color_features[idx*3+1]=result_green+0.5;
    color_features[idx*3+2]=result_blue+0.5;
    }

extern "C" __global__ void backward_sh(float* campos,float3* positions, float* sh, 
    float* sph_gauss_features, float* bandwidth_sharpness, float* lobe_axis, unsigned int num_sph_gauss,
    int sh_degree, int num_points,
    float3* dL_dRGB, float* dL_dsh, float3* dL_dmean,
    float* dL_dsph_gauss, float* dL_dbandwidth_sharpness, float* dL_dlobe_axis){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    //Check if gradient is zero
    if(dL_dRGB[idx].x==0 && dL_dRGB[idx].y==0 && dL_dRGB[idx].z==0){
        return;
    }
    float3 dir_orig = positions[idx] - make_float3(campos[0],campos[1],campos[2]);
    float3 dirs = dir_orig / length(dir_orig);
    
    int num_sh= (sh_degree+1)*(sh_degree+1);
    float x,y,z;
    x = dirs.x;
    y = dirs.y;
    z = dirs.z;
    
    float dL_dred=dL_dRGB[idx].x;
    float dL_dgreen=dL_dRGB[idx].y;
    float dL_dblue=dL_dRGB[idx].z;

    int index_red= idx*num_sh*3;
    int index_green= idx*num_sh*3+num_sh;
    int index_blue= idx*num_sh*3+2*num_sh;

    float3 dRGB_dx=make_float3(0,0,0);
    float3 dRGB_dy=make_float3(0,0,0);
    float3 dRGB_dz=make_float3(0,0,0);

    dL_dsh[index_red] = C0*dL_dred;
    dL_dsh[index_green] = C0*dL_dgreen;
    dL_dsh[index_blue] = C0*dL_dblue;
    if(sh_degree>0){
        float3 sh_1 = make_float3(sh[index_red+1],sh[index_green+1],sh[index_blue+1]);
        float3 sh_2 = make_float3(sh[index_red+2],sh[index_green+2],sh[index_blue+2]);
        float3 sh_3 = make_float3(sh[index_red+3],sh[index_green+3],sh[index_blue+3]);

        dL_dsh[index_red+1] = -C1*y*dL_dred;
        dL_dsh[index_green+1] = -C1*y*dL_dgreen;
        dL_dsh[index_blue+1] = -C1*y*dL_dblue;

        dL_dsh[index_red+2] = C1*z*dL_dred;
        dL_dsh[index_green+2] = C1*z*dL_dgreen;
        dL_dsh[index_blue+2] = C1*z*dL_dblue;

        dL_dsh[index_red+3] = -C1*x*dL_dred;
        dL_dsh[index_green+3] = -C1*x*dL_dgreen;
        dL_dsh[index_blue+3] = -C1*x*dL_dblue;


        dRGB_dx=-C1*sh_3;
        dRGB_dy=-C1*sh_1;
        dRGB_dz=C1*sh_2;
        if(sh_degree>1){
            float3 sh_4 = make_float3(sh[index_red+4],sh[index_green+4],sh[index_blue+4]);
            float3 sh_5 = make_float3(sh[index_red+5],sh[index_green+5],sh[index_blue+5]);
            float3 sh_6 = make_float3(sh[index_red+6],sh[index_green+6],sh[index_blue+6]);
            float3 sh_7 = make_float3(sh[index_red+7],sh[index_green+7],sh[index_blue+7]);
            float3 sh_8 = make_float3(sh[index_red+8],sh[index_green+8],sh[index_blue+8]);

            float xx = x*x;
            float yy = y*y;
            float zz = z*z;
            float xy = x*y;
            float yz = y*z;
            float xz = x*z;

            dL_dsh[index_red+4] = C2[0]*xy*dL_dred;
            dL_dsh[index_green+4] = C2[0]*xy*dL_dgreen;
            dL_dsh[index_blue+4] = C2[0]*xy*dL_dblue;

            dL_dsh[index_red+5] = C2[1]*yz*dL_dred;
            dL_dsh[index_green+5] = C2[1]*yz*dL_dgreen;
            dL_dsh[index_blue+5] = C2[1]*yz*dL_dblue;

            dL_dsh[index_red+6] = C2[2]*(2*zz-xx-yy)*dL_dred;
            dL_dsh[index_green+6] = C2[2]*(2*zz-xx-yy)*dL_dgreen;
            dL_dsh[index_blue+6] = C2[2]*(2*zz-xx-yy)*dL_dblue;

            dL_dsh[index_red+7] = C2[3]*xz*dL_dred;
            dL_dsh[index_green+7] = C2[3]*xz*dL_dgreen;
            dL_dsh[index_blue+7] = C2[3]*xz*dL_dblue;

            dL_dsh[index_red+8] = C2[4]*(xx-yy)*dL_dred;
            dL_dsh[index_green+8] = C2[4]*(xx-yy)*dL_dgreen;
            dL_dsh[index_blue+8] = C2[4]*(xx-yy)*dL_dblue;

            dRGB_dx+=C2[0]*y*sh_4+C2[2]*2.0*-x*sh_6+C2[3]*z*sh_7+C2[4]*2.0*x*sh_8;
            dRGB_dy+=C2[0]*x*sh_4+C2[1]*z*sh_5+C2[2]*2.0*-y*sh_6+C2[4]*2.0*-y*sh_8;
            dRGB_dz+=C2[1]*y*sh_5+C2[2]*2.0*2.0*z*sh_6+C2[3]*x*sh_7;

            if(sh_degree>2){
                float3 sh_9 = make_float3(sh[index_red+9],sh[index_green+9],sh[index_blue+9]);
                float3 sh_10 = make_float3(sh[index_red+10],sh[index_green+10],sh[index_blue+10]);
                float3 sh_11 = make_float3(sh[index_red+11],sh[index_green+11],sh[index_blue+11]);
                float3 sh_12 = make_float3(sh[index_red+12],sh[index_green+12],sh[index_blue+12]);
                float3 sh_13 = make_float3(sh[index_red+13],sh[index_green+13],sh[index_blue+13]);
                float3 sh_14 = make_float3(sh[index_red+14],sh[index_green+14],sh[index_blue+14]);
                float3 sh_15 = make_float3(sh[index_red+15],sh[index_green+15],sh[index_blue+15]);

                dL_dsh[index_red+9] = C3[0]*y*(3.0*xx-yy)*dL_dred;
                dL_dsh[index_green+9] = C3[0]*y*(3.0*xx-yy)*dL_dgreen;
                dL_dsh[index_blue+9] = C3[0]*y*(3.0*xx-yy)*dL_dblue;

                dL_dsh[index_red+10] = C3[1]*xy*z*dL_dred;
                dL_dsh[index_green+10] = C3[1]*xy*z*dL_dgreen;
                dL_dsh[index_blue+10] = C3[1]*xy*z*dL_dblue;

                dL_dsh[index_red+11] = C3[2]*y*(4.0*zz-xx-yy)*dL_dred;
                dL_dsh[index_green+11] = C3[2]*y*(4.0*zz-xx-yy)*dL_dgreen;
                dL_dsh[index_blue+11] = C3[2]*y*(4.0*zz-xx-yy)*dL_dblue;

                dL_dsh[index_red+12] = C3[3]*z*(2.0*zz-3.0*xx-3.0*yy)*dL_dred;
                dL_dsh[index_green+12] = C3[3]*z*(2.0*zz-3.0*xx-3.0*yy)*dL_dgreen;
                dL_dsh[index_blue+12] = C3[3]*z*(2.0*zz-3.0*xx-3.0*yy)*dL_dblue;

                dL_dsh[index_red+13] = C3[4]*x*(4.0*zz-xx-yy)*dL_dred;
                dL_dsh[index_green+13] = C3[4]*x*(4.0*zz-xx-yy)*dL_dgreen;
                dL_dsh[index_blue+13] = C3[4]*x*(4.0*zz-xx-yy)*dL_dblue;

                dL_dsh[index_red+14] = C3[5]*z*(xx-yy)*dL_dred;
                dL_dsh[index_green+14] = C3[5]*z*(xx-yy)*dL_dgreen;
                dL_dsh[index_blue+14] = C3[5]*z*(xx-yy)*dL_dblue;

                dL_dsh[index_red+15] = C3[6]*x*(xx-3.0*yy)*dL_dred;
                dL_dsh[index_green+15] = C3[6]*x*(xx-3.0*yy)*dL_dgreen;
                dL_dsh[index_blue+15] = C3[6]*x*(xx-3.0*yy)*dL_dblue;

                // dRGBdx += (
                    // C3[0] * sh[..., 9] * 3.0 * 2.0 * xy[:,None] +
                    // C3[1] * sh[..., 10] * yz[:,None] +
                    // C3[2] * sh[..., 11] * -2.0 * xy[:,None] +
                    // C3[3] * sh[..., 12] * -3.0 * 2.0 * xz[:,None] +
                    // C3[4] * sh[..., 13] * (-3.0 * xx[:,None] + 4.0 * zz[:,None] - yy[:,None]) +
                    // C3[5] * sh[..., 14] * 2.0 * xz[:,None] +
                    // C3[6] * sh[..., 15] * 3.0 * (xx[:,None] - yy[:,None]))
                dRGB_dx+=C3[0]*sh_9*3.0*2.0*xy+C3[1]*sh_10*yz+C3[2]*sh_11*-2.0*xy+C3[3]*sh_12*-3.0*2.0*xz+
                    C3[4]*sh_13*(-3.0*xx+4.0*zz-yy)+C3[5]*sh_14*2.0*xz+C3[6]*sh_15*3.0*(xx-yy);
                // dRGBdy += (
                //     C3[0] * sh[..., 9] * 3.0 * (xx[:,None] - yy[:,None]) +
                //     C3[1] * sh[..., 10] * xz[:,None] +
                //     C3[2] * sh[..., 11] * (-3.0 * yy[:,None] + 4.0 * zz[:,None] - xx[:,None]) +
                //     C3[3] * sh[..., 12] * -3.0 * 2.0 * yz[:,None] +
                //     C3[4] * sh[..., 13] * -2.0 * xy[:,None] +
                //     C3[5] * sh[..., 14] * -2.0 * yz[:,None] +
                //     C3[6] * sh[..., 15] * -3.0 * 2.0 * xy[:,None])
                dRGB_dy+=C3[0]*sh_9*(3.0*xx-yy)+C3[1]*sh_10*xz+C3[2]*sh_11*(-3.0*yy+4.0*zz-xx)+C3[3]*sh_12*-3.0*2.0*yz+
                    C3[4]*sh_13*-2.0*xy+C3[5]*sh_14*-2.0*yz+C3[6]*sh_15*-3.0*2.0*xy;
                // dRGBdz += (
                //     C3[1] * sh[..., 10] * xy[:,None] +
                //     C3[2] * sh[..., 11] * 4.0 * 2.0 * yz[:,None] +
                //     C3[3] * sh[..., 12] * 3.0 * (2.0 * zz[:,None] - xx[:,None] - yy[:,None]) +
                //     C3[4] * sh[..., 13] * 4.0 * 2.0 * xz[:,None] +
                //     C3[5] * sh[..., 14] * (xx[:,None] - yy[:,None]))
                dRGB_dz+=C3[1]*sh_10*xy+C3[2]*sh_11*4.0*2.0*yz+C3[3]*sh_12*3.0*(2.0*zz-xx-yy)+C3[4]*sh_13*4.0*2.0*xz+
                    C3[5]*sh_14*(xx-yy);

            }

        }
    }
    float3 dL_ddir = make_float3(dot(dL_dRGB[idx],dRGB_dx),dot(dL_dRGB[idx],dRGB_dy),dot(dL_dRGB[idx],dRGB_dz));

 
    unsigned int idx0 = idx*num_sph_gauss*3;
    for (int i = 0; i < num_sph_gauss; i++){
        float3 sph_gauss_feat_ = make_float3(sph_gauss_features[idx0+i],sph_gauss_features[idx0+i+num_sph_gauss],sph_gauss_features[idx0+i+2*num_sph_gauss]);
        float3 lobe_axis_ = make_float3(lobe_axis[idx0+i*3],lobe_axis[idx0+i*3+1],lobe_axis[idx0+i*3+2]);
        float sharpness = bandwidth_sharpness[idx*num_sph_gauss+i];
        float dot_dir_lobe_minus1 = dot(dirs,lobe_axis_)-1.0f;
        float sph_gauss_weight=exp((dot_dir_lobe_minus1)*sharpness);

        dL_dsph_gauss[idx0+i]=dL_dRGB[idx].x*sph_gauss_weight;
        dL_dsph_gauss[idx0+i+num_sph_gauss]=dL_dRGB[idx].y*sph_gauss_weight;
        dL_dsph_gauss[idx0+i+2*num_sph_gauss]=dL_dRGB[idx].z*sph_gauss_weight;
   
        float dL_dsph_gauss_weight = dot(dL_dRGB[idx],sph_gauss_feat_);

        dL_dlobe_axis[idx0+i*3]=dL_dsph_gauss_weight*sharpness*dirs.x*sph_gauss_weight;
        dL_dlobe_axis[idx0+i*3+1]=dL_dsph_gauss_weight*sharpness*dirs.y*sph_gauss_weight;
        dL_dlobe_axis[idx0+i*3+2]=dL_dsph_gauss_weight*sharpness*dirs.z*sph_gauss_weight;
        dL_dbandwidth_sharpness[idx*num_sph_gauss+i]=dL_dsph_gauss_weight*dot_dir_lobe_minus1*sph_gauss_weight;

        dL_ddir+=dL_dsph_gauss_weight*sharpness*sph_gauss_weight*lobe_axis_;
    }
    float sum2 = dot(dir_orig,dir_orig);
    float invsum32 = 1.0 / (sqrt(sum2 * sum2 * sum2));
    dL_dmean[idx] = make_float3(((sum2 - dir_orig.x * dir_orig.x) * dL_ddir.x - dir_orig.y * dir_orig.x * dL_ddir.y - dir_orig.z * dir_orig.x * dL_ddir.z) * invsum32,
                                (-dir_orig.x * dir_orig.y * dL_ddir.x + (sum2 - dir_orig.y * dir_orig.y) * dL_ddir.y - dir_orig.z * dir_orig.y * dL_ddir.z) * invsum32,
                                (-dir_orig.x * dir_orig.z * dL_ddir.x - dir_orig.y * dir_orig.z * dL_ddir.y + (sum2 - dir_orig.z * dir_orig.z) * dL_ddir.z) * invsum32);
    //Compute the gradients for the spherical gaussian parameters
}