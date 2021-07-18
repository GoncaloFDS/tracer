use crate::render::{
    acceleration_structures::{
        AccelerationStructureBuildGeometryInfo, AccelerationStructureGeometry,
        AccelerationStructureGeometryInfo, AccelerationStructureInfo, AccelerationStructureLevel,
    },
    buffer::{BufferInfo, BufferRegion},
    debug::DebugMessenger,
    device::Device,
    encoder::Encoder,
    instance,
    physical_device::PhysicalDevice,
    pipeline::PathTracingPipeline,
    pipeline::Pipeline,
    render_context::RenderContext,
    resources::{AccelerationStructure, Buffer},
    surface::Surface,
    swapchain::Swapchain,
};
use bevy::prelude::*;
use bumpalo::Bump;
use crevice::std430::AsStd430;
use erupt::{vk, EntryLoader, InstanceLoader};
use glam::vec3;
use parking_lot::Mutex;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;
use winit::window::Window;

pub struct Renderer {
    surface: Surface,
    swapchain: Swapchain,
    debug_messenger: DebugMessenger,
    physical_device: PhysicalDevice,
    render_context: RenderContext,
    path_tracing_pipeline: PathTracingPipeline,
    blases: HashMap<u8, AccelerationStructure>,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    blas_scratch: Option<Buffer>,
    bump: Mutex<Bump>,
    instance: Arc<InstanceLoader>,
    entry: EntryLoader,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = EntryLoader::new().unwrap();
        let instance = Arc::new(instance::create_instance(window, &entry));
        let debug_messenger = DebugMessenger::new(&instance);
        let surface = Surface::new(&instance, window);

        let device_extensions = vec![
            vk::KHR_SWAPCHAIN_EXTENSION_NAME,
            vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        ];
        let physical_device = PhysicalDevice::select_one(&instance, &surface, &device_extensions);
        let (device, queue) = physical_device.create_device(instance.clone(), &device_extensions);
        let render_context = RenderContext::new(device, queue);

        let mut swapchain = render_context.create_swapchain(&surface);
        swapchain.configure(&render_context.device, physical_device.info());

        let bump = Mutex::new(Bump::with_capacity(10000));
        let blases: HashMap<u8, AccelerationStructure> = Default::default();

        let path_tracing_pipeline = PathTracingPipeline::new(
            &render_context,
            physical_device.info().surface_format.format,
            physical_device.info().surface_capabilities.current_extent,
        );

        Renderer {
            surface,
            swapchain,
            debug_messenger,
            physical_device,
            render_context,
            path_tracing_pipeline,
            blases,
            vertex_buffer: None,
            index_buffer: None,
            blas_scratch: None,
            bump,
            instance,
            entry,
        }
    }

    pub fn draw(&mut self, camera: &GlobalTransform) {
        let mut encoder = self.render_context.queue.create_enconder();

        if let Entry::Vacant(entry) = self.blases.entry(0) {
            let vertices = [
                Vertex {
                    position: vec3(-0.5, -0.5, 0.0).into(),
                },
                Vertex {
                    position: vec3(0.5, -0.5, 0.0).into(),
                },
                Vertex {
                    position: vec3(0.0, 0.5, 0.0).into(),
                },
            ];
            let indices = [0u16, 1, 2];
            let bump = self.bump.lock();
            let (blas, vertex, index, scratch) = build_triangle_blas(
                &self.render_context,
                &mut encoder,
                &vertices,
                &indices,
                &bump,
            );
            self.vertex_buffer = Some(vertex);
            self.index_buffer = Some(index);
            self.blas_scratch = Some(scratch);
            entry.insert(blas);
            self.render_context
                .queue
                .submit(encoder.finish(&self.render_context), &[], &[], None);
        }

        let swapchain_image = loop {
            if let Some(swapchain_image) = self
                .swapchain
                .acquire_next_image(&self.render_context.device)
            {
                break swapchain_image;
            }
            self.swapchain
                .configure(&self.render_context.device, self.physical_device.info());
        };

        self.path_tracing_pipeline.draw(
            &mut self.render_context,
            swapchain_image.info().image.clone(),
            &swapchain_image.info().wait,
            &swapchain_image.info().signal,
            &self.blases,
            &self.bump.lock(),
            camera,
        );

        self.render_context.queue.present(swapchain_image);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.render_context.destroy_context();
            self.instance
                .destroy_surface_khr(Some(self.surface.handle()), None);
            self.debug_messenger.destroy(&self.instance);
            self.instance.destroy_instance(None);
        }
    }
}

fn build_triangle_blas<'a>(
    device: &Device,
    encoder: &mut Encoder<'a>,
    vertices: &[Vertex],
    indices: &[u16],
    bump: &'a Bump,
) -> (AccelerationStructure, Buffer, Buffer, Buffer) {
    let vertex_count = vertices.len();
    let vertex_stride = std::mem::size_of::<Std430Vertex>();
    let vertex_buffer_size = vertex_stride * vertex_count;
    let vertex_buffer = device.create_buffer_with_data(
        BufferInfo {
            align: 255,
            size: vertex_buffer_size as _,
            usage_flags: vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS
                | gpu_alloc::UsageFlags::HOST_ACCESS,
        },
        &vertices,
    );

    let triangle_count = indices.len() / 3;

    let index_count = indices.len();
    let index_buffer_size = std::mem::size_of::<u16>() * index_count;
    let index_buffer = device.create_buffer_with_data(
        BufferInfo {
            align: 255,
            size: index_buffer_size as _,
            usage_flags: vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS
                | gpu_alloc::UsageFlags::HOST_ACCESS,
        },
        &indices,
    );

    //
    let sizes = device.get_acceleration_structure_build_sizes(
        AccelerationStructureLevel::Bottom,
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR,
        &[AccelerationStructureGeometryInfo::Triangles {
            max_primitive_count: triangle_count as _,
            max_vertex_count: vertex_count as _,
            vertex_format: vk::Format::R32G32B32_SFLOAT,
            index_type: vk::IndexType::UINT16,
        }],
    );

    let blas_buffer = device.create_buffer(BufferInfo {
        align: 255,
        size: sizes.acceleration_structure_size,
        usage_flags: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        allocation_flags: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
    });

    let blas = device.create_acceleration_structure(AccelerationStructureInfo {
        level: AccelerationStructureLevel::Bottom,
        region: BufferRegion {
            buffer: blas_buffer,
            offset: 0,
            size: sizes.acceleration_structure_size,
            stride: None,
        },
    });

    let scratch = device.create_buffer(BufferInfo {
        align: 255,
        size: sizes.build_scratch_size,
        usage_flags: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS,
    });

    let geometries = bump.alloc([AccelerationStructureGeometry::Triangles {
        flags: vk::GeometryFlagsKHR::empty(),
        vertex_format: vk::Format::R32G32B32_SFLOAT,
        vertex_data: vertex_buffer.device_address().unwrap(),
        vertex_stride: vertex_stride as _,
        vertex_count: vertex_count as _,
        first_vertex: 0,
        primitive_count: triangle_count as _,
        index_data: index_buffer.device_address(),
        transform_data: None,
    }]);

    let build_info = bump.alloc([AccelerationStructureBuildGeometryInfo {
        src: None,
        dst: blas.clone(),
        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR,
        geometries,
        scratch: scratch.device_address().unwrap(),
    }]);

    encoder.build_acceleration_structure(build_info);

    (blas, vertex_buffer, index_buffer, scratch)
}

#[derive(Copy, Clone, Debug, AsStd430)]
struct Vertex {
    pub position: mint::Vector3<f32>,
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}
