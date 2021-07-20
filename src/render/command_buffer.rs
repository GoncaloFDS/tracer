use crate::render::acceleration_structures::IndexData;
use crate::render::{
    acceleration_structures::{
        AccelerationStructureBuildGeometryInfo, AccelerationStructureGeometry,
    },
    buffer::BufferRegion,
    device::Device,
    encoder::Command,
    image::ImageMemoryBarrier,
    pipeline::ShaderBindingTable,
    render_pass::{ClearValue, DEFAULT_ATTACHMENT_COUNT},
    resources::{
        Buffer, DescriptorSet, Framebuffer, GraphicsPipeline, PipelineLayout, RayTracingPipeline,
        RenderPass,
    },
    util::ToErupt,
};
use erupt::{vk, DeviceLoader};
use smallvec::SmallVec;
use std::ops::Range;

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    recording: bool,
}

impl CommandBuffer {
    pub fn new(handle: vk::CommandBuffer) -> Self {
        CommandBuffer {
            handle,
            recording: false,
        }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn write(&mut self, device: &Device, commands: &[Command<'_>]) {
        let device = device.handle();
        if !self.recording {
            unsafe {
                device
                    .begin_command_buffer(
                        self.handle,
                        &vk::CommandBufferBeginInfoBuilder::new()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap()
            }
            self.recording = true;
        }

        for command in commands {
            match *command {
                Command::BeginRenderPass {
                    render_pass,
                    framebuffer,
                    clears,
                } => self.begin_render_pass(device, &render_pass, &framebuffer, clears),
                Command::EndRenderPass => self.end_render_pass(device),
                Command::BindGraphicsPipeline { pipeline } => {
                    self.bind_graphics_pipeline(device, pipeline)
                }
                Command::BindRayTracingPipeline { pipeline } => {
                    self.bind_raytracing_pipeline(device, pipeline)
                }
                Command::BindDescriptorSets {
                    bind_point,
                    layout,
                    first_set,
                    descriptor_sets,
                    dynamic_offsets,
                } => self.bind_descriptor_sets(
                    device,
                    bind_point,
                    layout,
                    first_set,
                    &descriptor_sets,
                    dynamic_offsets,
                ),
                Command::SetViewport { viewport } => self.set_viewport(device, &viewport),
                Command::SetScissor { scissor } => self.set_scissor(device, &scissor),
                Command::Draw {
                    ref vertices,
                    ref instances,
                } => self.draw(device, vertices, instances),
                Command::DrawIndexed {
                    ref indices,
                    vertex_offset,
                    ref instances,
                } => self.draw_indexed(device, indices, vertex_offset, instances),
                Command::UpdateBuffer {
                    buffer,
                    offset,
                    data,
                } => self.update_buffer(device, buffer, offset, data),
                Command::BindVertexBuffers { first, buffers } => {
                    self.bind_vertex_buffers(device, first, buffers)
                }
                Command::BindIndexBuffer {
                    buffer,
                    offset,
                    index_type,
                } => self.bind_index_buffer(device, buffer, offset, index_type),
                Command::BuildAccelerationStructure { infos } => {
                    self.build_acceleration_structure(device, infos)
                }
                Command::TraceRays {
                    shader_binding_table,
                    extent,
                } => self.trace_rays(device, shader_binding_table, extent),
                Command::PipelineBarrier {
                    src,
                    dst,
                    src_access_mask,
                    dst_access_mask,
                    image_barriers,
                } => self.pipeline_barrier(
                    device,
                    src,
                    dst,
                    src_access_mask,
                    dst_access_mask,
                    image_barriers,
                ),
            }
        }

        unsafe {
            device.end_command_buffer(self.handle).unwrap();
        }
    }

    fn begin_render_pass(
        &mut self,
        device: &DeviceLoader,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        clears: &[ClearValue],
    ) {
        let mut clears = clears.into_iter();
        let clear_values = render_pass
            .info()
            .attachments
            .iter()
            .map(|_| {
                let clear = clears.next().unwrap();
                match *clear {
                    ClearValue::Color(r, g, b, a) => vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [r, g, b, a],
                        },
                    },
                    ClearValue::DepthStencil(depth, stencil) => vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                    },
                }
            })
            .collect::<SmallVec<[_; DEFAULT_ATTACHMENT_COUNT]>>();

        unsafe {
            device.cmd_begin_render_pass(
                self.handle,
                &vk::RenderPassBeginInfoBuilder::new()
                    .render_pass(render_pass.handle())
                    .framebuffer(framebuffer.handle())
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: framebuffer.info().extent,
                    })
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            )
        }
    }

    fn end_render_pass(&mut self, device: &DeviceLoader) {
        unsafe { device.cmd_end_render_pass(self.handle) }
    }

    fn bind_graphics_pipeline(&mut self, device: &DeviceLoader, pipeline: &GraphicsPipeline) {
        unsafe {
            device.cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle(),
            )
        }
    }

    fn bind_raytracing_pipeline(&mut self, device: &DeviceLoader, pipeline: &RayTracingPipeline) {
        unsafe {
            device.cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.handle(),
            )
        }
    }

    fn bind_descriptor_sets(
        &mut self,
        device: &DeviceLoader,
        bind_point: vk::PipelineBindPoint,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            device.cmd_bind_descriptor_sets(
                self.handle,
                bind_point,
                layout.handle(),
                first_set,
                &descriptor_sets
                    .iter()
                    .map(|set| set.handle())
                    .collect::<Vec<_>>(),
                dynamic_offsets,
            )
        }
    }

    fn set_viewport(&mut self, device: &DeviceLoader, viewport: &vk::Viewport) {
        unsafe { device.cmd_set_viewport(self.handle, 0, &[viewport.into_builder()]) }
    }

    fn set_scissor(&mut self, device: &DeviceLoader, scissor: &vk::Rect2D) {
        unsafe { device.cmd_set_scissor(self.handle, 0, &[scissor.into_builder()]) }
    }

    fn draw(&mut self, device: &DeviceLoader, vertices: &Range<u32>, instances: &Range<u32>) {
        unsafe {
            device.cmd_draw(
                self.handle,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
        }
    }

    fn draw_indexed(
        &mut self,
        device: &DeviceLoader,
        indices: &Range<u32>,
        vertex_offset: i32,
        instances: &Range<u32>,
    ) {
        unsafe {
            device.cmd_draw_indexed(
                self.handle,
                indices.end - indices.start,
                instances.end - indices.start,
                indices.start,
                vertex_offset,
                instances.start,
            )
        }
    }

    fn pipeline_barrier(
        &mut self,
        device: &DeviceLoader,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        image_barriers: &[ImageMemoryBarrier],
    ) {
        unsafe {
            device.cmd_pipeline_barrier(
                self.handle,
                src,
                dst,
                None,
                &[vk::MemoryBarrierBuilder::new()
                    .src_access_mask(src_access_mask)
                    .dst_access_mask(dst_access_mask)],
                &[],
                &image_barriers
                    .iter()
                    .map(|image_barrier| {
                        vk::ImageMemoryBarrierBuilder::new()
                            .image(image_barrier.image.handle())
                            .src_access_mask(src_access_mask)
                            .dst_access_mask(src_access_mask)
                            .old_layout(
                                image_barrier
                                    .old_layout
                                    .unwrap_or(vk::ImageLayout::UNDEFINED),
                            )
                            .new_layout(image_barrier.new_layout)
                            .src_queue_family_index(
                                image_barrier
                                    .family_transfer
                                    .as_ref()
                                    .map(|range| range.start)
                                    .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                            )
                            .dst_queue_family_index(
                                image_barrier
                                    .family_transfer
                                    .as_ref()
                                    .map(|range| range.end)
                                    .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                            )
                            .subresource_range(image_barrier.subresource.to_erupt())
                    })
                    .collect::<Vec<_>>(),
            )
        }
    }

    fn trace_rays(
        &mut self,
        device: &DeviceLoader,
        shader_binding_table: &ShaderBindingTable,
        extent: vk::Extent2D,
    ) {
        let to_erupt = |buffer_region: &BufferRegion| {
            let device_address = buffer_region.buffer.device_address().unwrap().0.get();

            vk::StridedDeviceAddressRegionKHRBuilder::new()
                .device_address(device_address + buffer_region.offset)
                .stride(buffer_region.stride.unwrap())
                .size(buffer_region.size)
                .build()
        };
        unsafe {
            device.cmd_trace_rays_khr(
                self.handle,
                &shader_binding_table
                    .raygen
                    .as_ref()
                    .map_or(vk::StridedDeviceAddressRegionKHR::default(), to_erupt),
                &shader_binding_table
                    .miss
                    .as_ref()
                    .map_or(vk::StridedDeviceAddressRegionKHR::default(), to_erupt),
                &shader_binding_table
                    .hit
                    .as_ref()
                    .map_or(vk::StridedDeviceAddressRegionKHR::default(), to_erupt),
                &shader_binding_table
                    .callable
                    .as_ref()
                    .map_or(vk::StridedDeviceAddressRegionKHR::default(), to_erupt),
                extent.width,
                extent.height,
                1,
            );
        }
    }

    fn build_acceleration_structure(
        &mut self,
        device: &DeviceLoader,
        infos: &[AccelerationStructureBuildGeometryInfo],
    ) {
        let mut geometries = vec![];
        let mut offsets = vec![];

        let ranges: Vec<_> = infos.iter().map(|info| {
            let offset = geometries.len();

            for geometry in info.geometries {
                match geometry {
                    AccelerationStructureGeometry::Triangles {
                        flags,
                        vertex_format,
                        vertex_data,
                        vertex_stride,
                        vertex_count,
                        first_vertex,
                        primitive_count,
                        index_data,
                        transform_data,
                    } => {
                        geometries.push(vk::AccelerationStructureGeometryKHRBuilder::new()
                            .flags(*flags)
                            .geometry_type(vk::GeometryTypeKHR::TRIANGLES_KHR)
                            .geometry(vk::AccelerationStructureGeometryDataKHR {
                                triangles: vk::AccelerationStructureGeometryTrianglesDataKHRBuilder::new()
                                    .vertex_format(*vertex_format)
                                    .vertex_data(vertex_data.to_erupt())
                                    .vertex_stride(*vertex_stride)
                                    .max_vertex(*vertex_count)
                                    .index_type(match index_data {
                                        None => vk::IndexType::NONE_KHR,
                                        Some(IndexData::U16(_)) => vk::IndexType::UINT16    ,
                                        Some(IndexData::U32(_)) => vk::IndexType::UINT32,
                                    })
                                    .index_data(match index_data {
                                        None => Default::default(),
                                        Some(IndexData::U16(device_address)) => device_address.to_erupt(),
                                        Some(IndexData::U32(device_address)) => device_address.to_erupt(),
                                    })
                                    .transform_data(transform_data.as_ref().map(|device_address| device_address.to_erupt()).unwrap_or_default())
                                    .build()
                            }));

                        offsets.push(vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
                            .primitive_count(*primitive_count)
                            .first_vertex(*first_vertex)
                        );
                    }
                    AccelerationStructureGeometry::Instances { flags, data, primitive_count } => {
                        geometries.push(vk::AccelerationStructureGeometryKHRBuilder::new()
                            .flags(flags.clone())
                            .geometry_type(vk::GeometryTypeKHR::INSTANCES_KHR)
                            .geometry(vk::AccelerationStructureGeometryDataKHR {
                                instances: vk::AccelerationStructureGeometryInstancesDataKHRBuilder::new()
                                    .data(data.to_erupt())
                                    .build()
                            }));

                        offsets.push(vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
                            .primitive_count(*primitive_count)
                        );
                    }
                }
            }

            offset..geometries.len()
        }).collect();

        let build_infos: SmallVec<[_; 32]> = infos
            .iter()
            .zip(&ranges)
            .map(|(info, range)| {
                let src = info
                    .src
                    .as_ref()
                    .map(|src| src.handle())
                    .unwrap_or_default();

                vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
                    ._type(info.dst.info().level.to_erupt())
                    .flags(info.flags.clone())
                    .mode(if info.src.is_some() {
                        vk::BuildAccelerationStructureModeKHR::UPDATE_KHR
                    } else {
                        vk::BuildAccelerationStructureModeKHR::BUILD_KHR
                    })
                    .src_acceleration_structure(src)
                    .dst_acceleration_structure(info.dst.handle())
                    .scratch_data(info.scratch.to_erupt())
                    .geometries(&geometries[range.clone()])
            })
            .collect();

        let build_offsets: SmallVec<[_; 32]> = ranges
            .into_iter()
            .map(|range| &*offsets[range][0] as *const _)
            .collect();

        unsafe {
            device.cmd_build_acceleration_structures_khr(self.handle, &build_infos, &build_offsets)
        }
    }

    fn bind_index_buffer(
        &mut self,
        device: &DeviceLoader,
        buffer: &Buffer,
        offset: u64,
        index_type: vk::IndexType,
    ) {
        unsafe { device.cmd_bind_index_buffer(self.handle, buffer.handle(), offset, index_type) }
    }

    fn bind_vertex_buffers(
        &mut self,
        device: &DeviceLoader,
        first: u32,
        buffers: &[(Buffer, u64)],
    ) {
        let (buffers, offsets): (Vec<_>, Vec<_>) = buffers
            .iter()
            .map(|(buffer, offset)| (buffer.handle(), offset))
            .unzip();
        unsafe { device.cmd_bind_vertex_buffers(self.handle, first, &buffers, &offsets) }
    }

    fn update_buffer(&mut self, device: &DeviceLoader, buffer: &Buffer, offset: u64, data: &[u8]) {
        unsafe {
            device.cmd_update_buffer(
                self.handle,
                buffer.handle(),
                offset,
                data.len() as _,
                data.as_ptr() as _,
            )
        }
    }
}
