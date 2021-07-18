use crate::render::pass::Pass;
use crate::render::{
    acceleration_structures::{
        AccelerationStructureBuildGeometryInfo, AccelerationStructureGeometry,
        AccelerationStructureGeometryInfo, AccelerationStructureInfo,
        AccelerationStructureInstance, AccelerationStructureLevel, TransformMatrix,
    },
    buffer::{BufferInfo, BufferRegion},
    descriptor::{
        DescriptorSetInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutInfo, DescriptorType,
        Descriptors, WriteDescriptorSet,
    },
    image::{Image, ImageInfo, ImageMemoryBarrier, ImageViewInfo},
    pipeline::{
        PipelineLayoutInfo, RayTracingPipelineInfo, RayTracingShaderGroupInfo, ShaderBindingTable,
        ShaderBindingTableInfo,
    },
    render_context::RenderContext,
    resources::{
        AccelerationStructure, Buffer, DescriptorSet, Fence, PipelineLayout, RayTracingPipeline,
        Semaphore,
    },
    shader::{Shader, ShaderModuleInfo},
};
use bevy::prelude::GlobalTransform;
use bumpalo::{collections::Vec as BVec, Bump};
use crevice::std430::{AsStd430, Std430};
use erupt::vk;
use glam::{vec3, vec4, Mat4, Vec3};
use std::collections::HashMap;

const MAX_INSTANCE_COUNT: u32 = 2048;

#[derive(AsStd430)]
pub struct Globals {
    camera: CameraUniform,
    color: mint::Vector4<f32>,
}

#[derive(AsStd430)]
pub struct CameraUniform {
    view: mint::ColumnMatrix4<f32>,
    proj: mint::ColumnMatrix4<f32>,
    view_inverse: mint::ColumnMatrix4<f32>,
    proj_inverse: mint::ColumnMatrix4<f32>,
}

pub struct RayTracingPass {
    pipeline_layout: PipelineLayout,
    pipeline: RayTracingPipeline,
    shader_binding_table: ShaderBindingTable,
    tlas: AccelerationStructure,
    scratch_buffer: Buffer,
    descriptor_set: DescriptorSet,
    globals: Globals,
    globals_buffer: Buffer,
    instances_buffer: Buffer,
    output_image: Image,
}

pub struct Input<'a> {
    pub blases: &'a HashMap<u8, AccelerationStructure>,
}

pub struct Output {
    pub tlas: AccelerationStructure,
    pub output_image: Image,
}

impl<'a> Pass<'a> for RayTracingPass {
    type Input = Input<'a>;
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
        let mut encoder = render_context.queue.create_enconder();

        let mut as_instances = BVec::new_in(bump);

        let blas = input.blases.get(&0u8).unwrap();

        as_instances.push(
            AccelerationStructureInstance::new(blas.device_address())
                .with_transform(TransformMatrix::identity()),
        );

        encoder.pipeline_barrier(
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            &[],
        );

        let view = Mat4::from_scale_rotation_translation(
            camera.scale,
            camera.rotation.inverse(),
            -camera.translation,
        );
        let proj = Mat4::perspective_rh(90.0f32.to_radians(), 800.0 / 600.0, 0.001, 10000.0);

        self.globals.camera = CameraUniform {
            view: view.into(),
            proj: proj.into(),
            view_inverse: view.inverse().into(),
            proj_inverse: proj.inverse().into(),
        };
        render_context.write_buffer(
            &mut self.globals_buffer,
            0,
            self.globals.as_std430().as_bytes(),
        );

        let build_info = bump.alloc([AccelerationStructureBuildGeometryInfo {
            src: None,
            dst: self.tlas.clone(),
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD_KHR,
            geometries: bump.alloc([AccelerationStructureGeometry::Instances {
                flags: vk::GeometryFlagsKHR::OPAQUE_KHR,
                data: self.instances_buffer.device_address().unwrap(),
                primitive_count: as_instances.len() as _,
            }]),
            scratch: self.scratch_buffer.device_address().unwrap(),
        }]);
        render_context.write_buffer(&mut self.instances_buffer, 0, &as_instances);

        encoder.build_acceleration_structure(build_info);

        encoder.bind_ray_tracing_pipeline(&self.pipeline);

        let descriptor_sets = [self.descriptor_set.clone()];
        encoder.bind_descriptor_sets(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            &self.pipeline_layout,
            0,
            &descriptor_sets,
            &[],
        );

        let image_barriers = [ImageMemoryBarrier::initialize_whole(
            &self.output_image,
            vk::ImageLayout::GENERAL,
        )];

        encoder.pipeline_barrier(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::MEMORY_WRITE,
            &image_barriers,
        );

        encoder.pipeline_barrier(
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::MEMORY_WRITE,
            &[],
        );

        encoder.trace_rays(&self.shader_binding_table, self.output_image.info().extent);

        let image_barriers = [ImageMemoryBarrier::transition_whole(
            &self.output_image,
            vk::ImageLayout::GENERAL..vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )];
        encoder.pipeline_barrier(
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::MEMORY_WRITE,
            &image_barriers,
        );

        let command_buffer = encoder.finish(&render_context.device);

        render_context
            .queue
            .submit(command_buffer, wait, signal, fence);

        Output {
            tlas: self.tlas.clone(),
            output_image: self.output_image.clone(),
        }
    }
}

impl RayTracingPass {
    pub fn new(render_context: &RenderContext, extent: vk::Extent2D) -> Self {
        let descriptor_set_layout =
            render_context.create_descriptor_set_layout(DescriptorSetLayoutInfo {
                bindings: vec![
                    // TLAS
                    DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: DescriptorType::AccelerationStructure,
                        count: 1,
                        stages: vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        flags: vk::DescriptorBindingFlags::empty(),
                    },
                    // Image
                    DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: DescriptorType::StorageImage,
                        count: 1,
                        stages: vk::ShaderStageFlags::RAYGEN_KHR,
                        flags: vk::DescriptorBindingFlags::empty(),
                    },
                    // Globals
                    DescriptorSetLayoutBinding {
                        binding: 2,
                        descriptor_type: DescriptorType::UniformBuffer,
                        count: 1,
                        stages: vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                            | vk::ShaderStageFlags::MISS_KHR,
                        flags: vk::DescriptorBindingFlags::empty(),
                    },
                ],
                flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            });

        let pipeline_layout = render_context.create_pipeline_layout(PipelineLayoutInfo {
            sets: vec![descriptor_set_layout.clone()],
            push_constants: vec![],
        });

        let ray_gen_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("raytrace.rgen.spv")),
            vk::ShaderStageFlagBits::RAYGEN_KHR,
        );

        let miss_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("raytrace.rmiss.spv")),
            vk::ShaderStageFlagBits::MISS_KHR,
        );

        let closest_hit_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("raytrace.rchit.spv")),
            vk::ShaderStageFlagBits::CLOSEST_HIT_KHR,
        );

        let pipeline = render_context.create_ray_tracing_pipeline(RayTracingPipelineInfo {
            shaders: vec![ray_gen_shader, miss_shader, closest_hit_shader],
            groups: vec![
                RayTracingShaderGroupInfo::Raygen { raygen: 0 },
                RayTracingShaderGroupInfo::Miss { miss: 1 },
                RayTracingShaderGroupInfo::Triangle {
                    any_hit: None,
                    closest_hit: Some(2),
                },
            ],
            max_recursion_depth: 2,
            layout: pipeline_layout.clone(),
        });

        let shader_binding_table = render_context.create_shader_binding_table(
            &pipeline,
            ShaderBindingTableInfo {
                raygen: Some(0),
                miss: &[1],
                hit: &[2],
                callable: &[],
            },
        );

        let tlas_build_sizes = render_context.get_acceleration_structure_build_sizes(
            AccelerationStructureLevel::Top,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD_KHR,
            &[AccelerationStructureGeometryInfo::Instances {
                max_primitive_count: MAX_INSTANCE_COUNT,
            }],
        );

        let tlas_buffer = render_context.create_buffer(BufferInfo {
            align: 255,
            size: tlas_build_sizes.acceleration_structure_size,
            usage_flags: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            allocation_flags: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
        });

        let tlas = render_context.create_acceleration_structure(AccelerationStructureInfo {
            level: AccelerationStructureLevel::Top,
            region: BufferRegion::whole(tlas_buffer),
        });

        let scratch_buffer = render_context.create_buffer(BufferInfo {
            align: 255,
            size: tlas_build_sizes.build_scratch_size,
            usage_flags: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS,
        });

        let instances_buffer = render_context.create_buffer(BufferInfo {
            align: 255,
            size: std::mem::size_of::<
                [vk::AccelerationStructureInstanceKHR; MAX_INSTANCE_COUNT as usize],
            >() as _,
            usage_flags: vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS
                | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                | gpu_alloc::UsageFlags::HOST_ACCESS,
        });

        let mut globals_buffer = render_context.create_buffer(BufferInfo {
            align: 255,
            size: std::mem::size_of::<Std430Globals>() as _,
            usage_flags: vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS
                | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                | gpu_alloc::UsageFlags::HOST_ACCESS,
        });

        let cam_pos = vec3(0.0, 1.0, -1.0);
        let view = Mat4::look_at_rh(cam_pos, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(70.0f32.to_radians(), 800.0 / 600.0, 0.001, 10000.0);

        let globals = Globals {
            color: vec4(0.8, 0.0, 0.0, 1.0).into(),
            camera: CameraUniform {
                view: view.into(),
                proj: proj.into(),
                view_inverse: view.inverse().into(),
                proj_inverse: proj.inverse().into(),
            },
        };
        render_context.write_buffer(&mut globals_buffer, 0, globals.as_std430().as_bytes());

        let output_image = render_context.create_image(ImageInfo {
            extent,
            format: vk::Format::R32G32B32A32_SFLOAT,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlagBits::_1,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        });

        let output_image_view = render_context.create_image_view(ImageViewInfo::new(
            output_image.clone(),
            vk::ImageAspectFlags::COLOR,
        ));

        let descriptor_set = render_context.create_descriptor_set(DescriptorSetInfo {
            layout: descriptor_set_layout.clone(),
        });

        render_context.update_descriptor_sets(
            &[
                WriteDescriptorSet {
                    descriptor_set: &descriptor_set,
                    binding: 0,
                    element: 0,
                    descriptors: Descriptors::AccelerationStructure(std::slice::from_ref(&tlas)),
                },
                WriteDescriptorSet {
                    descriptor_set: &descriptor_set,
                    binding: 1,
                    element: 0,
                    descriptors: Descriptors::StorageImage(&[(
                        output_image_view.clone(),
                        vk::ImageLayout::GENERAL,
                    )]),
                },
                WriteDescriptorSet {
                    descriptor_set: &descriptor_set,
                    binding: 2,
                    element: 0,
                    descriptors: Descriptors::UniformBuffer(&[(
                        globals_buffer.clone(),
                        0,
                        std::mem::size_of::<Std430Globals>() as _,
                    )]),
                },
            ],
            &[],
        );

        RayTracingPass {
            pipeline_layout,
            pipeline,
            shader_binding_table,
            tlas,
            scratch_buffer,
            descriptor_set,
            globals,
            globals_buffer,
            instances_buffer,
            output_image,
        }
    }
}
