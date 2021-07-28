use crate::render::buffer::BufferInfo;
use crate::render::descriptor::{Descriptors, WriteDescriptorSet};
use crate::render::framebuffer::FramebufferInfo;
use crate::render::image::{ImageInfo, ImageView, ImageViewInfo};
use crate::render::pipeline::{PushConstant, VertexInputAttribute, VertexInputBinding};
use crate::render::render_pass::ClearValue;
use crate::render::resources::{Buffer, Framebuffer, Sampler};
use crate::render::{
    descriptor::{
        DescriptorSetInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutInfo, DescriptorType,
    },
    image::Image,
    pass::Pass,
    pipeline::{GraphicsPipelineInfo, PipelineLayoutInfo, Rasterizer},
    render_context::RenderContext,
    render_pass::{AttachmentInfo, RenderPassInfo, Subpass},
    resources::{DescriptorSet, Fence, GraphicsPipeline, PipelineLayout, RenderPass, Semaphore},
    shader::{Shader, ShaderModuleInfo},
};
use bevy::prelude::GlobalTransform;
use bumpalo::{collections::Vec as BumpVec, Bump};
use egui::paint::ClippedShape;
use egui::{epaint, ClippedMesh, CtxRef, Pos2, RawInput, Rect, TextureId};
use erupt::vk;
use lru::LruCache;
use smallvec::smallvec;
use std::sync::Arc;

pub struct Input {
    pub target: Image,
}

pub struct Output;

pub struct UIPass {
    egui_context: CtxRef,
    raw_input: RawInput,

    render_pass: RenderPass,
    pipeline_layout: PipelineLayout,
    graphics_pipeline: GraphicsPipeline,

    framebuffers: LruCache<Image, Framebuffer>,

    descriptor_sets: [DescriptorSet; 2],
    vertex_buffers: [Buffer; 2],
    index_buffers: [Buffer; 2],

    font_sampler: Sampler,
    font_image: Option<Image>,

    clipped_meshes: Vec<egui::ClippedMesh>,
    texture_version: u64,
}

impl UIPass {
    pub fn new(
        render_context: &RenderContext,
        surface_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Self {
        let egui_context = CtxRef::default();

        let raw_input = RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::new(0.0, 0.0),
                egui::vec2(extent.width as f32, extent.height as f32),
            )),
            time: Some(0.0),
            ..Default::default()
        };

        let descriptor_set_layout =
            render_context.create_descriptor_set_layout(DescriptorSetLayoutInfo {
                bindings: vec![DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: DescriptorType::CombinedImageSampler,
                    count: 1,
                    stages: vk::ShaderStageFlags::FRAGMENT,
                    flags: Default::default(),
                }],
                flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
            });

        let vertex_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("ui.vert.spv")),
            vk::ShaderStageFlagBits::VERTEX,
        );

        let fragment_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("ui.frag.spv")),
            vk::ShaderStageFlagBits::FRAGMENT,
        );

        let render_pass = render_context.create_render_pass(RenderPassInfo {
            attachments: smallvec![AttachmentInfo {
                format: surface_format,
                samples: vk::SampleCountFlags::_1,
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::STORE,
                initial_layout: None,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            }],
            subpasses: smallvec![Subpass {
                colors: smallvec![0],
                depth: None,
            }],
        });

        let pipeline_layout = render_context.create_pipeline_layout(PipelineLayoutInfo {
            sets: vec![descriptor_set_layout.clone()],
            push_constants: vec![PushConstant {
                stages: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: 2 * std::mem::size_of::<f32>() as u32,
            }],
        });

        let graphics_pipeline = render_context.create_graphics_pipeline(GraphicsPipelineInfo {
            vertex_bindings: vec![VertexInputBinding {
                input_rate: vk::VertexInputRate::VERTEX,
                // epaint::Vertex stride
                stride: 5 * std::mem::size_of::<f32>() as u32,
            }],
            vertex_attributes: vec![
                // position
                VertexInputAttribute {
                    location: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    binding: 0,
                    offset: 0,
                },
                // uv
                VertexInputAttribute {
                    location: 1,
                    format: vk::Format::R32G32_SFLOAT,
                    binding: 0,
                    offset: 8,
                },
                // color
                VertexInputAttribute {
                    location: 2,
                    format: vk::Format::R8G8B8A8_UNORM,
                    binding: 0,
                    offset: 16,
                },
            ],
            primitive_topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            vertex_shader: vertex_shader.clone(),
            rasterizer: Some(Rasterizer {
                viewport: vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as _,
                    height: extent.height as _,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                depth_clamp: false,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                cull_mode: vk::CullModeFlags::NONE,
                polygon_mode: vk::PolygonMode::FILL,
                fragment_shader: Some(fragment_shader.clone()),
            }),
            layout: pipeline_layout.clone(),
            render_pass: render_pass.clone(),
            subpass: 0,
        });

        let descriptor_sets = [
            render_context.create_descriptor_set(DescriptorSetInfo {
                layout: descriptor_set_layout.clone(),
            }),
            render_context.create_descriptor_set(DescriptorSetInfo {
                layout: descriptor_set_layout.clone(),
            }),
        ];

        let vertex_buffers = [
            render_context.create_buffer(BufferInfo {
                align: 255,
                size: Self::vertex_buffer_size(),
                usage_flags: vk::BufferUsageFlags::VERTEX_BUFFER,
                allocation_flags: gpu_alloc::UsageFlags::empty(),
            }),
            render_context.create_buffer(BufferInfo {
                align: 255,
                size: Self::vertex_buffer_size(),
                usage_flags: vk::BufferUsageFlags::VERTEX_BUFFER,
                allocation_flags: gpu_alloc::UsageFlags::empty(),
            }),
        ];

        let index_buffers = [
            render_context.create_buffer(BufferInfo {
                align: 255,
                size: Self::index_buffer_size(),
                usage_flags: vk::BufferUsageFlags::INDEX_BUFFER,
                allocation_flags: gpu_alloc::UsageFlags::empty(),
            }),
            render_context.create_buffer(BufferInfo {
                align: 255,
                size: Self::index_buffer_size(),
                usage_flags: vk::BufferUsageFlags::INDEX_BUFFER,
                allocation_flags: gpu_alloc::UsageFlags::empty(),
            }),
        ];

        let sampler = render_context.create_sampler();

        UIPass {
            egui_context,
            raw_input,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            framebuffers: LruCache::new(4),
            descriptor_sets,
            vertex_buffers,
            index_buffers,
            font_sampler: sampler,
            font_image: None,
            clipped_meshes: vec![],
            texture_version: 0,
        }
    }

    fn vertex_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    fn index_buffer_size() -> u64 {
        1024 * 1024 * 2
    }

    pub fn context(&self) -> CtxRef {
        self.egui_context.clone()
    }

    pub fn begin_frame(&mut self) {
        self.egui_context.begin_frame(self.raw_input.take());
    }

    pub fn end_frame(&mut self) {
        let (_output, clipped_shapes) = self.egui_context.end_frame();
        self.clipped_meshes = self.egui_context.tessellate(clipped_shapes);
    }

    fn update_set(&mut self, render_context: &mut RenderContext, frame_id: usize) {
        let texture = self.egui_context.texture();
        if texture.version == self.texture_version {
            return;
        }
        self.texture_version = texture.version;
        let image = self.create_font_texture(texture, render_context);
        self.font_image = Some(image.clone());
        let image_view = render_context
            .create_image_view(ImageViewInfo::new(image, vk::ImageAspectFlags::COLOR));
        render_context.update_descriptor_sets(
            &[WriteDescriptorSet {
                descriptor_set: &self.descriptor_sets[frame_id],
                binding: 0,
                element: 0,
                descriptors: Descriptors::CombinedImageSampler(&[(
                    image_view,
                    vk::ImageLayout::GENERAL,
                    self.font_sampler.clone(),
                )]),
            }],
            &[],
        )
    }

    fn create_font_texture(
        &self,
        texture: Arc<epaint::Texture>,
        render_context: &mut RenderContext,
    ) -> Image {
        let image_data = &texture
            .pixels
            .iter()
            .flat_map(|&r| vec![r, r, r, r])
            .collect::<Vec<_>>();

        let image = render_context.create_image_with_data(
            ImageInfo {
                extent: vk::Extent2D {
                    width: texture.width as u32,
                    height: texture.height as u32,
                },
                format: vk::Format::R8G8B8A8_UNORM,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlagBits::_1,
                usage: vk::ImageUsageFlags::SAMPLED,
            },
            vk::ImageLayout::GENERAL,
            image_data,
        );

        image
    }
}

impl Pass<'_> for UIPass {
    type Input = Input;
    type Output = Output;

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
    ) -> Self::Output {
        let framebuffer = match self.framebuffers.get(&input.target) {
            None => {
                let color_view = render_context.create_image_view(ImageViewInfo::new(
                    input.target.clone(),
                    vk::ImageAspectFlags::COLOR,
                ));

                let framebuffer = render_context.create_framebuffer(FramebufferInfo {
                    render_pass: self.render_pass.clone(),
                    views: smallvec![color_view],
                    extent: input.target.info().extent,
                });

                self.framebuffers.put(input.target, framebuffer.clone());
                framebuffer
            }
            Some(framebuffer) => framebuffer.clone(),
        };

        let frame_id = (frame % 2) as usize;

        self.update_set(render_context, frame_id);

        let mut encoder = render_context.queue.create_enconder();

        encoder.begin_render_pass(
            &self.render_pass,
            &framebuffer,
            &[ClearValue::Color(0.5, 0.2, 0.2, 0.0)],
        );

        encoder.bind_graphics_pipeline(&self.graphics_pipeline);

        let mut to_bind = BumpVec::with_capacity_in(self.vertex_buffers.len(), bump);
        to_bind.push((self.vertex_buffers[frame_id].clone(), 0));

        encoder.bind_vertex_buffers(0, to_bind.into_bump_slice());

        encoder.bind_index_buffer(
            bump.alloc(self.index_buffers[frame_id].clone()),
            0,
            vk::IndexType::UINT32,
        );

        encoder.set_viewport(vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: framebuffer.info().extent.width as f32,
            height: framebuffer.info().extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        });

        encoder.set_scissor(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: framebuffer.info().extent,
        });

        let width = framebuffer.info().extent.width as f32;
        let height = framebuffer.info().extent.height as f32;
        let push = [width, height];
        encoder.push_constants(
            &self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &push,
        );

        for ClippedMesh(rect, mesh) in &self.clipped_meshes {
            if let TextureId::User(id) = mesh.texture_id {}
        }

        // encoder.draw_indexed(0..0, 0, 0..0);

        encoder.end_render_pass();

        let command_buffer = encoder.finish(&render_context.device);

        render_context
            .queue
            .submit(command_buffer, wait, signal, fence);

        Output
    }
}
