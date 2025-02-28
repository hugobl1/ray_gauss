#pragma once
#define BUFFER_SIZE 8
#define DT 0.0025f
#define TRANSMITTANCE_EPSILON 0.003f
#define SIGMA_THRESHOLD 0.1f

namespace sphere {
    const unsigned int NUM_ATTRIBUTE_VALUES = 4u;

    struct SphereHitGroupData {
        float3* positions;
        float3* scales;
        float4* quaternions;
    };
}

struct Params
{   
    unsigned int iteration;
    unsigned int jitter;
    unsigned int rnd_sample;
    unsigned int           max_prim_slice;

    float3 bbox_min;
    float3 bbox_max;

    unsigned int image_width;
    unsigned int image_height;

    float3 cam_eye;
    float3 cam_u;
    float3 cam_v;
    float3 cam_w;
    // float2 cam_tan_half_FOV;

    float cam_tan_half_fovx;
    float cam_tan_half_fovy;

    float* densities;
    float* color_features;
    float3* positions;
    float3* scales;
    float4* quaternions;

    int* hit_prim_idx;

    float3* ray_colors;

    OptixTraversableHandle handle;
    
    int* max_gaussians_exceeded;
    
    unsigned int white_background;

    // BASELINE
    float3* __restrict__ ray_bboxes;
    float3* __restrict__ ray_normals;
    float*  __restrict__ ray_densities;
    float*  __restrict__ ray_depths;
};


struct MissData
{
    float3 bg_color;
};


template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}


// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}
