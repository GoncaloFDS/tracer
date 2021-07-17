#version 460

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 output_color;

layout(binding = 0, set = 0) uniform sampler2D initial_image;


void main() {
    output_color = texture(initial_image, in_uv);
}
