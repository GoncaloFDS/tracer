#version 460

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 output_color;

layout(binding = 0, set = 0) uniform sampler2D initial_image;


void main() {
    float gamma = 1.0 / 2.2;
    output_color = pow(texture(initial_image, in_uv), vec4(gamma));
}
