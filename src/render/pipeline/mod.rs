pub use self::graphics_pipeline::*;
pub use self::raster_pipeline::*;
pub use self::ray_tracing_pipeline::*;

use crate::render::{
    image::Image,
    render_context::RenderContext,
    resources::{AccelerationStructure, DescriptorSetLayout, Semaphore},
};
use bevy::prelude::GlobalTransform;
use bumpalo::Bump;
use erupt::vk;
use std::collections::HashMap;

mod graphics_pipeline;
mod raster_pipeline;
mod ray_tracing_pipeline;

pub trait Pipeline {
    fn draw(
        &mut self,
        render_context: &mut RenderContext,
        target: Image,
        target_wait: &Semaphore,
        target_signal: &Semaphore,
        blases: &HashMap<u8, AccelerationStructure>,
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
