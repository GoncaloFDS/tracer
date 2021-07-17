#extension GL_EXT_scalar_block_layout : enable
#extension GL_ARB_gpu_shader_int64 : enable

struct Camera {
    mat4 view;
    mat4 proj;
    mat4 view_inverse;
    mat4 proj_inverse;
};

struct PerRayData {
    vec3 hit_color;
};
