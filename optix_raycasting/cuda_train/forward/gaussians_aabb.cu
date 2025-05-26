#include <optix.h>

#include "vec_math.h"
#include "helpers.h"
#include "gaussians_aabb.h"

#define float3_as_ints(u) float_as_int(u.x), float_as_int(u.y), float_as_int(u.z)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};

__device__ __constant__ float3 BBOX_COLORS[] = {
	float3{ 1.f, 0.f, 0.f },
    float3{ 0.f, 1.f, 0.f },
    float3{ 0.f, 0.f, 1.f },
    float3{ 1.f, 1.f, 0.f },
    float3{ 0.f, 1.f, 1.f },
    float3{ 1.f, 0.f, 1.f },
    float3{ 1.f, 1.f, 1.f },
};
constexpr __device__ unsigned int BBOX_COLORS_LEN = 7;

__device__ float3 computeColorFromSG_float3(int num_sph_gauss, const float3 gaussian_pos, const float3 campos, const float *sg_x, const float *sg_y, const float *sg_z,
                                            const float *bandwidth_sharpness, const float *lobe_axis)
{
    float3 dir = gaussian_pos - campos;
    dir = dir / length(dir);

    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    float result_x = 0.0f;
    float result_y = 0.0f;
    float result_z = 0.0f;

    for (int l = 0; l < num_sph_gauss; l++)
    {

        float x_ = sg_x[l];
        float y_ = sg_y[l];
        float z_ = sg_z[l];

        float sharpness = bandwidth_sharpness[l];
        float3 axis = make_float3(lobe_axis[l * 3], lobe_axis[l * 3 + 1], lobe_axis[l * 3 + 2]);

        float dot_product_axis = dot(axis, dir);
        float gaussian = expf(sharpness * (dot_product_axis - 1.0f));

        result_x += gaussian * x_;
        result_y += gaussian * y_;
        result_z += gaussian * z_;
    }
    return make_float3(result_x, result_y, result_z);
}

__device__ float3 computeColorFromSH_float3(int deg, const float3 gaussian_pos, const float3 campos, const float *sh_x, const float *sh_y, const float *sh_z)
{
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    // glm::vec3 pos = means[idx];

    float3 dir = gaussian_pos - campos;
    dir = dir / length(dir);

    float result_x = SH_C0 * sh_x[0];
    float result_y = SH_C0 * sh_y[0];
    float result_z = SH_C0 * sh_z[0];
    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        // result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        result_x = result_x - SH_C1 * y * sh_x[1] + SH_C1 * z * sh_x[2] - SH_C1 * x * sh_x[3];
        result_y = result_y - SH_C1 * y * sh_y[1] + SH_C1 * z * sh_y[2] - SH_C1 * x * sh_y[3];
        result_z = result_z - SH_C1 * y * sh_z[1] + SH_C1 * z * sh_z[2] - SH_C1 * x * sh_z[3];
        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            // result = result +
            // 	SH_C2[0] * xy * sh[4] +
            // 	SH_C2[1] * yz * sh[5] +
            // 	SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
            // 	SH_C2[3] * xz * sh[7] +
            // 	SH_C2[4] * (xx - yy) * sh[8];
            result_x = result_x +
                       SH_C2[0] * xy * sh_x[4] +
                       SH_C2[1] * yz * sh_x[5] +
                       SH_C2[2] * (2.0f * zz - xx - yy) * sh_x[6] +
                       SH_C2[3] * xz * sh_x[7] +
                       SH_C2[4] * (xx - yy) * sh_x[8];
            result_y = result_y +
                       SH_C2[0] * xy * sh_y[4] +
                       SH_C2[1] * yz * sh_y[5] +
                       SH_C2[2] * (2.0f * zz - xx - yy) * sh_y[6] +
                       SH_C2[3] * xz * sh_y[7] +
                       SH_C2[4] * (xx - yy) * sh_y[8];
            result_z = result_z +
                       SH_C2[0] * xy * sh_z[4] +
                       SH_C2[1] * yz * sh_z[5] +
                       SH_C2[2] * (2.0f * zz - xx - yy) * sh_z[6] +
                       SH_C2[3] * xz * sh_z[7] +
                       SH_C2[4] * (xx - yy) * sh_z[8];
            if (deg > 2)
            {
                // result = result +
                // 	SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                // 	SH_C3[1] * xy * z * sh[10] +
                // 	SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                // 	SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                // 	SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                // 	SH_C3[5] * z * (xx - yy) * sh[14] +
                // 	SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
                result_x = result_x +
                           SH_C3[0] * y * (3.0f * xx - yy) * sh_x[9] +
                           SH_C3[1] * xy * z * sh_x[10] +
                           SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_x[11] +
                           SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_x[12] +
                           SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_x[13] +
                           SH_C3[5] * z * (xx - yy) * sh_x[14] +
                           SH_C3[6] * x * (xx - 3.0f * yy) * sh_x[15];
                result_y = result_y +
                           SH_C3[0] * y * (3.0f * xx - yy) * sh_y[9] +
                           SH_C3[1] * xy * z * sh_y[10] +
                           SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_y[11] +
                           SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_y[12] +
                           SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_y[13] +
                           SH_C3[5] * z * (xx - yy) * sh_y[14] +
                           SH_C3[6] * x * (xx - 3.0f * yy) * sh_y[15];
                result_z = result_z +
                           SH_C3[0] * y * (3.0f * xx - yy) * sh_z[9] +
                           SH_C3[1] * xy * z * sh_z[10] +
                           SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_z[11] +
                           SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_z[12] +
                           SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_z[13] +
                           SH_C3[5] * z * (xx - yy) * sh_z[14] +
                           SH_C3[6] * x * (xx - 3.0f * yy) * sh_z[15];
            }
        }
    }
    // result += 0.5f;
    // result_x += 0.5f;
    // result_y += 0.5f;
    // result_z += 0.5f;
    // result = fmax(result, 0.0f);
    return make_float3(result_x, result_y, result_z);
}

__device__ float computeColorFromSH(int deg, const float3 gaussian_pos, const float3 campos, const float *sh)
{
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    // glm::vec3 pos = means[idx];

    float3 dir = gaussian_pos - campos;
    dir = dir / length(dir);

    float result = SH_C0 * sh[0];

    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                     SH_C2[0] * xy * sh[4] +
                     SH_C2[1] * yz * sh[5] +
                     SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                     SH_C2[3] * xz * sh[7] +
                     SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2)
            {
                result = result +
                         SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                         SH_C3[1] * xy * z * sh[10] +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                         SH_C3[5] * z * (xx - yy) * sh[14] +
                         SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;
    // result = fmax(result, 0.0f);
    return result;
}

__device__ void evalShBases(unsigned int deg, float3 ray_direction, float *spherical_harmonics_bases)
{
    // Evaluate spherical harmonics bases at unit directions,
    // without taking linear combination.
    // At each point, the final result may the be
    // obtained through simple multiplication.

    // :param deg: int SH max degree. Currently, 0-4 supported
    // :param ray_direction: torch.Tensor (..., 3) unit directions

    // :return: float array (..., (deg + 1) ** 2) SH bases

    // Check that deg is between 0 and 4
    float C0 = 0.28209479177387814f;
    float C1 = 0.4886025119029199f;
    float C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
    float C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};
    float C4[9] = {2.5033429417967046f, -1.7701307697799304f, 0.9461746957575601f, -0.6690465435572892f, 0.10578554691520431f, -0.6690465435572892f, 0.47308734787878004f, -1.7701307697799304f, 0.6258357354491761f};
    if (deg < 0 || deg > 4)
    {
        printf("The degree of the spherical harmonics must be between 0 and 4");
        return;
    }
    spherical_harmonics_bases[0] = C0;
    if (deg > 0)
    {
        float x = ray_direction.x;
        float y = ray_direction.y;
        float z = ray_direction.z;
        spherical_harmonics_bases[1] = -C1 * y;
        spherical_harmonics_bases[2] = C1 * z;
        spherical_harmonics_bases[3] = -C1 * x;
        if (deg > 1)
        {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;
            spherical_harmonics_bases[4] = C2[0] * xy;
            spherical_harmonics_bases[5] = C2[1] * yz;
            spherical_harmonics_bases[6] = C2[2] * (2.0f * zz - xx - yy);
            spherical_harmonics_bases[7] = C2[3] * xz;
            spherical_harmonics_bases[8] = C2[4] * (xx - yy);
            if (deg > 2)
            {
                spherical_harmonics_bases[9] = C3[0] * y * (3.0f * xx - yy);
                spherical_harmonics_bases[10] = C3[1] * xy * z;
                spherical_harmonics_bases[11] = C3[2] * y * (4.0f * zz - xx - yy);
                spherical_harmonics_bases[12] = C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
                spherical_harmonics_bases[13] = C3[4] * x * (4.0f * zz - xx - yy);
                spherical_harmonics_bases[14] = C3[5] * z * (xx - yy);
                spherical_harmonics_bases[15] = C3[6] * x * (xx - 3.0f * yy);
                // spherical_harmonics_bases[10]=C3[1]*z*(4*zz-xx-yy);
                // spherical_harmonics_bases[11]=C3[2]*x*(4*xx-yy-zz);
                // spherical_harmonics_bases[12]=C3[3]*yz*(4*zz-xx-yy);
                // spherical_harmonics_bases[13]=C3[4]*xz*(4*zz-xx-yy);
                // spherical_harmonics_bases[14]=C3[5]*x*(xx-yy);
                // spherical_harmonics_bases[15]=C3[6]*y*(yy-xx);
                if (deg > 3)
                {
                    spherical_harmonics_bases[16] = C4[0] * y * z * (6.0f * zz - xx - yy);
                    spherical_harmonics_bases[17] = C4[1] * y * z * (3.0f * xx - yy);
                    spherical_harmonics_bases[18] = C4[2] * z * (4.0f * zz - xx - yy) * (xx - yy);
                    spherical_harmonics_bases[19] = C4[3] * x * z * (3.0f * xx - yy);
                    spherical_harmonics_bases[20] = C4[4] * x * z * (xx - yy);
                    spherical_harmonics_bases[21] = C4[5] * x * y * (6.0f * xx - yy - zz);
                    spherical_harmonics_bases[22] = C4[6] * x * y * (xx - yy);
                    spherical_harmonics_bases[23] = C4[7] * z * (4.0f * zz - xx - yy) * (xx - yy);
                    spherical_harmonics_bases[24] = C4[8] * y * (xx - yy) * (xx - yy);
                }
            }
        }
    }
}

// __forceinline__ __device__ void quaternion_to_matrix(const float4& q, float3& col0, float3& col1, float3& col2){
// 	float r = q.x;
// 	float x = q.y;
// 	float y = q.z;
// 	float z = q.w;
//     col0=make_float3(1.0f-2.0f*(y*y+z*z),
//                     2.f * (x * y + r * z),
//                     2.f * (x * z - r * y));
//     col1=make_float3(2.f * (x * y - r * z),
//                     1.0f-2.0f*(x*x+z*z),
//                     2.f * (y * z + r * x));
//     col2=make_float3(2.f * (x * z + r * y),
//                     2.f * (y * z - r * x),
//                     1.0f-2.0f*(x*x+y*y));
// }

__forceinline__ __device__ void quaternion_to_matrix(const float4 &q, float3 &col0, float3 &col1, float3 &col2)
{
    float r = q.x;
    float i = q.y;
    float j = q.z;
    float k = q.w;
    col0 = make_float3(1.0f - 2.0f * (j * j + k * k),
                       2.f * (i * j + r * k),
                       2.f * (i * k - r * j));
    col1 = make_float3(2.f * (i * j - r * k),
                       1.0f - 2.0f * (i * i + k * k),
                       2.f * (j * k + r * i));
    col2 = make_float3(2.f * (i * k + r * j),
                       2.f * (j * k - r * i),
                       1.0f - 2.0f * (i * i + j * j));
}

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

extern "C"
{
    __constant__ Params params;
}

extern "C" __global__ void __intersection__gaussian()
{
    const sphere::SphereHitGroupData *hit_group_data = reinterpret_cast<sphere::SphereHitGroupData *>(optixGetSbtDataPointer());

    const unsigned int primitive_index = optixGetPrimitiveIndex();

    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    const float3 gaussian_positions = hit_group_data->positions[primitive_index];
    float3 scales = hit_group_data->scales[primitive_index];
    float gaussian_density = params.densities[primitive_index];
    // float sigm_alpha = (1/(1+expf(-gaussian_density)));
    // float density_threshold = 1.0f;

    float ratio = gaussian_density / SIGMA_THRESHOLD;
    scales = scales * sqrtf(logf(ratio * ratio));

    float3 inv_scales = make_float3(1.0f / scales.x, 1.0f / scales.y, 1.0f / scales.z);

    float4 quaternion = hit_group_data->quaternions[primitive_index];
    float3 U_rot, V_rot, W_rot;
    quaternion_to_matrix(quaternion, U_rot, V_rot, W_rot);

    // M=(M1,M2,M3) = (RS^{-1})^T where S is the scaling matrix and R the rotation matrix so Sigma^{-1}=M^T*M
    float3 M1, M2, M3;
    U_rot = U_rot * inv_scales.x;
    V_rot = V_rot * inv_scales.y;
    W_rot = W_rot * inv_scales.z;
    M1 = make_float3(U_rot.x, V_rot.x, W_rot.x);
    M2 = make_float3(U_rot.y, V_rot.y, W_rot.y);
    M3 = make_float3(U_rot.z, V_rot.z, W_rot.z);

    const float3 O = ray_origin - gaussian_positions;
    // const float3 O_ellipsis=make_float3(O.x/scales.x,O.y/scales.y,O.z/scales.z);
    const float3 O_ellipsis = M1 * O.x + M2 * O.y + M3 * O.z;
    // const float  l      = 1.0f / length( ray_direction );
    // const float3 D      = ray_direction * l;
    // const float3 ray_direction_ellipsis_ref=make_float3(ray_direction.x/scales.x,ray_direction.y/scales.y,ray_direction.z/scales.z);
    // const float3 dir_ellipsis=make_float3(ray_direction.x/scales.x,ray_direction.y/scales.y,ray_direction.z/scales.z);
    const float3 dir_ellipsis = M1 * ray_direction.x + M2 * ray_direction.y + M3 * ray_direction.z;
    const float l = 1.0f / length(dir_ellipsis);
    const float3 D_ellipsis = dir_ellipsis * l;
    float b = -dot(O_ellipsis, D_ellipsis);
    float c = dot(O_ellipsis, O_ellipsis) - 1;
    float dists_projection_point_squared = (dot(O_ellipsis + b * D_ellipsis, O_ellipsis + b * D_ellipsis));
    float disc = 1 - dists_projection_point_squared;

    if (disc > 1e-7f)
    {
        float sdisc = sqrtf(disc);
        int sign_b = (b > 0) ? 1 : -1;
        float q = b + sign_b * sdisc;
        float root1 = (c / q) * l;
        float root2 = q * l;

        // V2
        //  if(primitive_index==12312){
        //      float a_bis=dot(dir_ellipsis,dir_ellipsis);
        //      float b_bis=-dot(O_ellipsis,dir_ellipsis);
        //      float disc_bis=1-(dot(O_ellipsis+(b_bis/a_bis)*dir_ellipsis,O_ellipsis+(b_bis/a_bis)*dir_ellipsis));
        //      float c_bis=dot(O_ellipsis,O_ellipsis)-1;
        //      float q_bis=b_bis+sign_b*sqrt(disc_bis*a_bis);
        //      float t0_bis=(c_bis/q_bis);
        //      float t1_bis=(q_bis/a_bis);
        //      printf("root1 %f, root2 %f, t0_bis %f, t1_bis %f\n",root1,root2,t0_bis,t1_bis);
        //  }

        float min_t = fmaxf(ray_tmin, root1);
        float max_t = fminf(ray_tmax, root2);

        if ((min_t <= max_t))
        {
            optixReportIntersection(ray_tmax, 0);
        }
    }
}

static __forceinline__ __device__ void computeRay(unsigned int idx_ray, float3 &origin, float3 &direction, unsigned int &seed)
{
    // #ifdef JITTER
    float idx_x, idx_y;
    if (params.jitter == 1)
    {
        // subpixel_jitter is a random variable between 0 and 1
        // unsigned int t0 = clock();
        // unsigned int seed = tea<4>( idx_ray, t0 );
        const float jitter_x = rnd(seed) * 0.333333f;
        const float jitter_y = rnd(seed) * 0.333333f;
        const float2 subpixel_jitter = make_float2(jitter_x, jitter_y);

        idx_x = (idx_ray % params.image_width + 0.333333f + subpixel_jitter.x) / params.image_width;
        idx_y = (idx_ray / params.image_width + 0.333333f + subpixel_jitter.y) / params.image_height;
    }
    else
    {
        idx_x = (idx_ray % params.image_width + 0.5f) / params.image_width;
        idx_y = (idx_ray / params.image_width + 0.5f) / params.image_height;
    }
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;

    const float2 pixel_idx = 2.0f * make_float2(idx_x, idx_y) - 1.0f;
    const float2 cam_tan_half_fov = make_float2(params.cam_tan_half_fovx, params.cam_tan_half_fovy);
    const float2 d = pixel_idx * cam_tan_half_fov;
    origin = params.cam_eye;
    direction = normalize(d.x * U + d.y * V + W);
}


typedef struct Quaternion
{
    float w, x, y, z;

    __forceinline__ Quaternion operator*( Quaternion other )
    {
        return Quaternion
        {
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w - y * other.z + z * other.y,
            w * other.y + x * other.z + y * other.w - z * other.x,
            w * other.z - x * other.y + y * other.x + z * other.w
        };
    }

    __forceinline__ Quaternion operator*( float3 other )
    {
        return Quaternion
        {
            w * 0.0f    - x * other.x - y * other.y - z * other.z,
            w * other.x + x * 0.0f    - y * other.z + z * other.y,
            w * other.y + x * other.z + y * 0.0f    - z * other.x,
            w * other.z - x * other.y + y * other.x + z * 0.0f   
        };
    }
} Quaternion;

__forceinline__ __device__ Quaternion quaternion_inverse(const Quaternion& q)
{
    return Quaternion { q.w, -q.x, -q.y, -q.z };
}

template <typename T> /* , typename std::enable_if<
    std::is_same<T, Quaternion>::value || std::is_same<T, float3>::value, int>::type = 0> */
__forceinline__ __device__ Quaternion quaternion_rotate(const Quaternion& q, const T& p)
{
    return ( quaternion_inverse(q) * p ) * q;
}

static __forceinline__ __device__ void computeBufferForward(const unsigned int idx_ray, const unsigned int p0, const float dt, const float tbuffer,
                                                            const float3 ray_origin, const float3 ray_direction,
                                                            float *density_buffer, float3 *color_buffer, float *bbox_density_buffer, float3 *bbox_buffer, unsigned int &seed,
                                                            float* depth_buffer, float3* normal_buffer)
{
    for (int prim_iter = 0; prim_iter < p0; prim_iter++)
    {
        unsigned int current_seed = seed;
        int primitive_index = params.hit_prim_idx[idx_ray * params.max_prim_slice + prim_iter];
        const float3 gaussian_pos = params.positions[primitive_index];
        float3 scales = params.scales[primitive_index];
        float3 inv_scales = make_float3(1.0f / scales.x, 1.0f / scales.y, 1.0f / scales.z);
        float4 quaternion = params.quaternions[primitive_index];
        float3 U_rot, V_rot, W_rot;
        quaternion_to_matrix(quaternion, U_rot, V_rot, W_rot);

        // M=(M1,M2,M3) = (RS^{-1})^T where S is the scaling matrix and R the rotation matrix so Sigma^{-1}=M^T*M
        float3 M1, M2, M3;
        U_rot = U_rot * inv_scales.x;
        V_rot = V_rot * inv_scales.y;
        W_rot = W_rot * inv_scales.z;
        M1 = make_float3(U_rot.x, V_rot.x, W_rot.x);
        M2 = make_float3(U_rot.y, V_rot.y, W_rot.y);
        M3 = make_float3(U_rot.z, V_rot.z, W_rot.z);

        // float3 M_d=ray_direction.x*M1+ray_direction.y*M2+ray_direction.z*M3;

        // float3 gaussian_color = make_float3(0.0f);
        // gaussian_color=computeColorFromSH_float3(degree_sh, gaussian_pos, params.cam_eye, params.color_features+primitive_index*num_sh*3,params.color_features+primitive_index*num_sh*3+num_sh,params.color_features+primitive_index*num_sh*3+num_sh*2);

        // gaussian_color+=make_float3(0.5f,0.5f,0.5f);

        // gaussian_color+=computeColorFromSG_float3(params.num_sph_gauss, gaussian_pos, params.cam_eye, params.sph_gauss_features+primitive_index*params.num_sph_gauss*3,
        //     params.sph_gauss_features+primitive_index*params.num_sph_gauss*3+params.num_sph_gauss,params.sph_gauss_features+primitive_index*params.num_sph_gauss*3+2*params.num_sph_gauss,
        //     params.bandwidth_sharpness+primitive_index*params.num_sph_gauss,params.lobe_axis+primitive_index*3*params.num_sph_gauss);

        float3 gaussian_color = make_float3(params.color_features[primitive_index * 3], params.color_features[primitive_index * 3 + 1], params.color_features[primitive_index * 3 + 2]);
        for (int index_buffer = 0; index_buffer < BUFFER_SIZE; index_buffer++)
        {
            float t_sample;
            if (params.rnd_sample == 0)
            {
                t_sample = tbuffer + index_buffer * dt;
            }
            else if (params.rnd_sample == 1)
            {
                t_sample = tbuffer + index_buffer * dt + rnd(current_seed) * dt;
            }
            else
            {
                t_sample = tbuffer + index_buffer * dt + 0.5f * dt;
            }
            float3 hit_sample = ray_origin + ray_direction * t_sample;
            float3 xhit_xgaus = hit_sample - gaussian_pos;
            float3 M_xhit_xgaus = xhit_xgaus.x * M1 + xhit_xgaus.y * M2 + xhit_xgaus.z * M3;
            float power = -0.5f * dot(M_xhit_xgaus, M_xhit_xgaus);
            float weight_density = expf(power);
            float gaussian_density = params.densities[primitive_index];

            if (gaussian_density * weight_density > SIGMA_THRESHOLD)
            {
                density_buffer[index_buffer] += gaussian_density * weight_density;
                color_buffer[index_buffer]   += gaussian_color * weight_density * gaussian_density;
                // BASELINE
                bbox_buffer[index_buffer]    += gaussian_density * BBOX_COLORS[ primitive_index % BBOX_COLORS_LEN ];
                depth_buffer[index_buffer]   += gaussian_density * weight_density * t_sample;
                // normal based on shortest axis
                int shortest_axis = scales.x < scales.z ? (int)(scales.x >= scales.y) : (scales.z < scales.y ? 2 : 1);
                float _ax[3] = {0,0,0};
                _ax[shortest_axis] = 1.0f;
                float3 ax = *reinterpret_cast<float3*>(_ax);

                const Quaternion q = *reinterpret_cast<Quaternion*>(&quaternion);
                Quaternion normal = quaternion_rotate( q, ax );
                normal_buffer[index_buffer]  += gaussian_density * weight_density * make_float3( normal.x, normal.y, normal.z );
            }
        }
        if (prim_iter == p0 - 1)
        {
            seed = current_seed;
        }
    }
}

static __forceinline__ __device__ void colorBlending(const unsigned int idx_ray, const unsigned int p0, const float dt, const float tbuffer,
                                                     const float3 ray_origin, const float3 ray_direction,
                                                     float *density_buffer, float3 *color_buffer, float3 *bbox_buffer, float3 &ray_color, float &transmittance, unsigned int &seed,
                                                     float3 &bbox_color, float &B_T,  float3 &ray_normal, float& ray_depth)
{
    float bbox_density_buffer[BUFFER_SIZE] = {0.0f};
    float depth_buffer[BUFFER_SIZE] = {0.0f};
    float3 normal_buffer[BUFFER_SIZE] = {0.0f};

    computeBufferForward(idx_ray, p0, dt, tbuffer,
                         ray_origin, ray_direction,
                         density_buffer, color_buffer, bbox_density_buffer, bbox_buffer, seed,
                         depth_buffer, normal_buffer);
    
    for (int index_buffer = 0; index_buffer < BUFFER_SIZE; index_buffer++)
    {
        float buffer_density = density_buffer[index_buffer];
        float alpha = 1.0f - exp(-buffer_density * dt);
        float3 buffer_color = color_buffer[index_buffer];
        float3 buffer_color_normalized = buffer_color;
        if (buffer_density > 0.0f)
        {
            buffer_color_normalized /= buffer_density;
        }

        ray_color += transmittance * alpha * buffer_color_normalized;

        // BASELINE 
        float3 box_col = bbox_buffer[index_buffer];
        float gaussian_density = bbox_density_buffer[index_buffer]; // technically the accumulated density

        float3 bbox_color_normalized = box_col;
        if (buffer_density > 0.0f)
        {
            bbox_color_normalized /= gaussian_density;
        }
        bbox_color += B_T * bbox_color_normalized;
        B_T        *= 1.0f - (gaussian_density * dt);

        if (buffer_density <= 0.0f)
            buffer_density = 1.0f;

        ray_depth  += transmittance * alpha * (depth_buffer[index_buffer] / buffer_density);
        ray_normal += transmittance * alpha * (normal_buffer[index_buffer] / buffer_density);

        transmittance *= 1.0f - alpha;

    }
}

extern "C" __global__ void __raygen__rg()
{

    const unsigned int idx_ray = optixGetLaunchIndex().x;

    unsigned int seed = tea<4>(params.iteration, idx_ray);

    float3 ray_origin, ray_direction;
    computeRay(idx_ray, ray_origin, ray_direction, seed);
    // printf("idx_ray %d, ray_origin %f %f %f, ray_direction %f %f %f\n",idx_ray,ray_origin.x,ray_origin.y,ray_origin.z,ray_direction.x,ray_direction.y,ray_direction.z);

    const float3 bbox_min = params.bbox_min;
    const float3 bbox_max = params.bbox_max;

    float3 t0, t1, tmin, tmax;
    t0 = (bbox_min - ray_origin) / ray_direction;
    t1 = (bbox_max - ray_origin) / ray_direction;
    tmin = fminf(t0, t1);
    tmax = fmaxf(t0, t1);
    float tenter = fmaxf(0.0f, fmaxf(tmin.x, fmaxf(tmin.y, tmin.z)));
    float texit = fminf(tmax.x, fminf(tmax.y, tmax.z));

    // float big_diag = length(bbox_max - bbox_min);
    // float slab_spacing = (texit - tenter) / 10.0f;
    // const float slab_spacing = 0.02; //length(bbox_max - bbox_min) / (1024.0f/BUFFER_SIZE) ;
    // const float dt=slab_spacing/BUFFER_SIZE;
    const float dt = DT;
    const float slab_spacing = dt * BUFFER_SIZE;

    float transmittance = 1.0f;
    float3 ray_color = make_float3(0.0f);

    // BASELINE
    float B_T = 1.0f;
    float3 bbox_color = make_float3(0.0f);
    float3 ray_normal = make_float3(0.0f);
    float ray_depth = 0.0;

    if (tenter < texit)
    {
        // float tbuffer=0.0f;
        float tbuffer = tenter;
        float t_min_slab;
        float t_max_slab;
        unsigned int p0 = 0;
        unsigned int bool_not_access;

        while (tbuffer < texit && transmittance > TRANSMITTANCE_EPSILON)
        {

            p0 = 0;

            t_min_slab = fmaxf(tenter, tbuffer);
            t_max_slab = fminf(texit, tbuffer + slab_spacing);
            if (t_max_slab > tenter)
            {

                optixTrace(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    t_min_slab,
                    t_max_slab,
                    0.0f, // rayTime
                    OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_NONE,
                    0, // SBT offset
                    0, // SBT stride
                    0, // missSBTIndex
                    p0);

                if (p0 == 0)
                {
                    tbuffer += slab_spacing;
                    continue;
                }

                float density_buffer[BUFFER_SIZE] = {0.0f};
                float3 color_buffer[BUFFER_SIZE] = {make_float3(0.0f)};
                float3 bbox_buffer[BUFFER_SIZE] = {make_float3(0.0f)};
                colorBlending(idx_ray, p0, dt, tbuffer,
                              ray_origin, ray_direction,
                              density_buffer, color_buffer, bbox_buffer, ray_color, transmittance, seed, 
                              bbox_color, B_T, ray_normal, ray_depth);
            }
            tbuffer += slab_spacing;
        }
        float3 bg_color = make_float3(1.0f, 1.0f, 1.0f);
        if (params.white_background == 0)
        {
            bg_color = make_float3(0.0f, 0.0f, 0.0f);
        }
        params.ray_colors[idx_ray]      = ray_color + transmittance * bg_color;
        // BASELINE
        params.ray_bboxes[idx_ray]      = bbox_color + B_T * bg_color;

        if ( (abs(ray_normal.x) + abs(ray_normal.y) + abs(ray_normal.z)) > 0)
            ray_normal /= norm3df( ray_normal.x, ray_normal.y, ray_normal.z );
        else
            ray_normal = bg_color;
        params.ray_normals[idx_ray]     = (1.0f-transmittance) * ray_normal + transmittance * bg_color;

        params.ray_depths[idx_ray]      = ray_depth;
        params.ray_densities[idx_ray]   = 1.0f - transmittance;
    }
    else
    {
        float3 bg_color = make_float3(1.0f, 1.0f, 1.0f);
        if (params.white_background == 0)
        {
            bg_color = make_float3(0.0f, 0.0f, 0.0f);
        }
        params.ray_colors[idx_ray] = bg_color;
        // BASELINE
        params.ray_bboxes[idx_ray]      = bg_color;
        params.ray_normals[idx_ray]     = bg_color;
        params.ray_depths[idx_ray]      = 13;
        params.ray_densities[idx_ray]   = 0.0f;
    }
}

extern "C" __global__ void __miss__ms()
{
}

extern "C" __global__ void __anyhit__ah()
{
    const unsigned int num_primitives = optixGetPayload_0();

    if (num_primitives >= params.max_prim_slice)
    {
        // printf("The number of spheres is greater than the maximum number of spheres per ray\n");
        atomicAdd(&(params.max_gaussians_exceeded[0]), 1);
        optixTerminateRay();
        return;
    }

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int idx_ray = idx.x;
    const unsigned int current_sphere_idx = optixGetPrimitiveIndex();

    params.hit_prim_idx[idx_ray * params.max_prim_slice + num_primitives] = current_sphere_idx;

    optixSetPayload_0(num_primitives + 1);
    optixIgnoreIntersection();
}