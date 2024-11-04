import cupy as cp
#Define absolute difference kernel

kernel_code = """
extern "C" __global__
void abs_diff(uchar4* a, uchar3* b, int size, bool log_scale) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        unsigned int abs_red, abs_green, abs_blue;
        float abs_mean;
        abs_red=abs(a[idx].x - b[idx].x);
        abs_green=abs(a[idx].y - b[idx].y);
        abs_blue=abs(a[idx].z - b[idx].z);
        abs_mean=(abs_red+abs_green+abs_blue)/3;
        if(log_scale){
            // log scaling for value between [0,255]
            float log_abs_mean=log(abs_mean+1.0f);
            log_abs_mean=log_abs_mean/log(256.0f);
            log_abs_mean*=255;
            abs_mean=log_abs_mean;
        }
        a[idx].x=abs_mean;
        a[idx].y=abs_mean;
        a[idx].z=abs_mean;
    }
}
"""
# Compile the kernel
abs_diff_kernel = cp.RawKernel(kernel_code, 'abs_diff')

kernel_code_copy = """
extern "C" __global__
void identity(uchar4* a, uchar3* b, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        a[idx].x = b[idx].x;
        a[idx].y = b[idx].y;
        a[idx].z = b[idx].z;
        a[idx].w = 255;
    }
}
"""
copy_kernel = cp.RawKernel(kernel_code_copy, 'identity')


kernel_code_copy_normalize_depth = """
extern "C" __global__
void copy_normalize_depth(uchar4* a, float* b, float min_depth,float max_depth, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        float depth=b[idx];
        float depth_normalized=(depth-min_depth)/(max_depth-min_depth);
        a[idx].x = depth_normalized*255;
        a[idx].y = depth_normalized*255;
        a[idx].z = depth_normalized*255;
        a[idx].w = 255;
    }
}
"""

copy_normalize_depth_kernel = cp.RawKernel(kernel_code_copy_normalize_depth, 'copy_normalize_depth')