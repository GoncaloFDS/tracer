pub mod raster_pass;
pub mod raytracing_pass;
pub mod tonemap_pass;
pub mod ui_pass;

use crate::render::{
    render_context::RenderContext,
    resources::{Fence, Semaphore},
};
use bevy::prelude::GlobalTransform;
use bumpalo::Bump;
use erupt::vk;
pub use raster_pass::*;

pub trait Pass<'a> {
    type Input;
    type Output;

    fn draw(
        &mut self,
        input: Self::Input,
        frame: u64,
        wait: &[(vk::PipelineStageFlags, Semaphore)],
        signal: &[Semaphore],
        fence: Option<&Fence>,
        render_context: &mut RenderContext,
        bump: &Bump,
        camera: &GlobalTransform,
    ) -> Self::Output;
}
