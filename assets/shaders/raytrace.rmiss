#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "common/descriptors.glsl"

layout(location = 0) rayPayloadInEXT PerRayData prd;

void main() {
    prd.hit_color = vec3(0.2, 0.2, 0.5);
}
