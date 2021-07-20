use crate::render::{
    debug::DebugMessenger,
    instance,
    mesh::Mesh,
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
use erupt::{vk, EntryLoader, InstanceLoader};
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
    blases: HashMap<Handle<Mesh>, AccelerationStructure>,
    vertex_buffer: HashMap<Handle<Mesh>, Buffer>,
    index_buffer: HashMap<Handle<Mesh>, Buffer>,
    blas_scratch: HashMap<Handle<Mesh>, Buffer>,
    bump: Mutex<Bump>,
    instance: Arc<InstanceLoader>,
    _entry: EntryLoader,
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
            blases: Default::default(),
            vertex_buffer: Default::default(),
            index_buffer: Default::default(),
            blas_scratch: Default::default(),
            bump,
            instance,
            _entry: entry,
        }
    }

    pub fn load_models(&mut self, handle: &Handle<Mesh>, mesh: &Mesh) {
        let mut encoder = self.render_context.queue.create_enconder();
        if let Entry::Vacant(entry) = self.blases.entry(handle.clone()) {
            let bump = self.bump.lock();

            let (blas, vertex, index, scratch) =
                mesh.build_triangle_blas(&self.render_context, &mut encoder, &bump);
            self.vertex_buffer.insert(handle.clone(), vertex);
            self.index_buffer.insert(handle.clone(), index);
            self.blas_scratch.insert(handle.clone(), scratch);
            entry.insert(blas);
            self.render_context
                .queue
                .submit(encoder.finish(&self.render_context), &[], &[], None);
        }
    }

    pub fn draw(&mut self, camera: &GlobalTransform) {
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
