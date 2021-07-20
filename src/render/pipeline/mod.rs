pub use self::graphics_pipeline::*;
pub use self::ray_tracing_pipeline::*;

use crate::render::mesh::Mesh;
use crate::render::{
    image::Image,
    render_context::RenderContext,
    resources::{AccelerationStructure, DescriptorSetLayout, Semaphore},
};
use bevy::asset::Handle;
use bevy::prelude::GlobalTransform;
use bumpalo::Bump;
use erupt::vk;
use std::collections::HashMap;

mod graphics_pipeline;
mod ray_tracing_pipeline;
pub mod vertex_format;

pub trait Pipeline {
    fn draw(
        &mut self,
        render_context: &mut RenderContext,
        target: Image,
        target_wait: &Semaphore,
        target_signal: &Semaphore,
        blases: &HashMap<Handle<Mesh>, AccelerationStructure>,
        bump: &Bump,
        camera: &GlobalTransform,
    );
}

#[derive(Clone)]
pub struct PipelineLayoutInfo {
    pub sets: Vec<DescriptorSetLayout>,
    pub push_constants: Vec<PushConstant>,
}

#[derive(Clone)]
pub struct PushConstant {
    pub stages: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}
