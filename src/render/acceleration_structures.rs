use crate::render::{
    buffer::{BufferRegion, DeviceAddress},
    resources::AccelerationStructure,
    util::ToErupt,
};
use erupt::vk;

#[derive(Clone)]
pub struct AccelerationStructureInfo {
    pub level: AccelerationStructureLevel,
    pub region: BufferRegion,
}

#[derive(Clone)]
pub struct AccelerationStructureBuildSizesInfo {
    pub acceleration_structure_size: u64,
    pub update_scratch_size: u64,
    pub build_scratch_size: u64,
}

#[derive(Clone)]
pub enum AccelerationStructureLevel {
    Bottom,
    Top,
}

impl ToErupt<vk::AccelerationStructureTypeKHR> for AccelerationStructureLevel {
    fn to_erupt(&self) -> vk::AccelerationStructureTypeKHR {
        match self {
            AccelerationStructureLevel::Bottom => {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR
            }
            AccelerationStructureLevel::Top => vk::AccelerationStructureTypeKHR::TOP_LEVEL_KHR,
        }
    }
}

#[derive(Clone)]
pub enum AccelerationStructureGeometryInfo {
    Triangles {
        max_primitive_count: u32,
        max_vertex_count: u32,
        vertex_format: vk::Format,
        index_type: vk::IndexType,
    },
    Instances {
        max_primitive_count: u32,
    },
}

#[derive(Clone)]
pub struct AccelerationStructureBuildGeometryInfo<'a> {
    pub src: Option<AccelerationStructure>,
    pub dst: AccelerationStructure,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub geometries: &'a [AccelerationStructureGeometry],
    pub scratch: DeviceAddress,
}

pub enum AccelerationStructureGeometry {
    Triangles {
        flags: vk::GeometryFlagsKHR,
        vertex_format: vk::Format,
        vertex_data: DeviceAddress,
        vertex_stride: u64,
        vertex_count: u32,
        first_vertex: u32,
        primitive_count: u32,
        index_data: Option<DeviceAddress>,
        transform_data: Option<DeviceAddress>,
    },
    Instances {
        flags: vk::GeometryFlagsKHR,
        data: DeviceAddress,
        primitive_count: u32,
    },
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TransformMatrix {
    pub matrix: [[f32; 4]; 3],
}

impl TransformMatrix {
    pub fn identity() -> Self {
        TransformMatrix {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
    }
}

impl Default for TransformMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct InstanceCustomIndexAndMask(pub u32);

impl InstanceCustomIndexAndMask {
    pub fn new(custom_index: u32, mask: u8) -> Self {
        assert!(custom_index < 1u32 << 24);

        InstanceCustomIndexAndMask(custom_index | ((mask as u32) << 24))
    }
}

impl Default for InstanceCustomIndexAndMask {
    fn default() -> Self {
        InstanceCustomIndexAndMask::new(0, !0)
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(align(16))]
#[repr(C)]
pub struct AccelerationStructureInstance {
    pub transform: TransformMatrix,
    pub custom_index_mask: InstanceCustomIndexAndMask,
    pub shader_binding_offset_flags: InstanceShaderBindingOffsetAndFlags,
    pub acceleration_structure_reference: DeviceAddress,
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct InstanceShaderBindingOffsetAndFlags(pub u32);

impl InstanceShaderBindingOffsetAndFlags {
    pub fn new(instance_shader_binding_offset: u32, flags: vk::GeometryInstanceFlagsKHR) -> Self {
        assert!(instance_shader_binding_offset < 1u32 << 24);

        InstanceShaderBindingOffsetAndFlags(
            instance_shader_binding_offset | ((flags.bits() as u32) << 24),
        )
    }
}

impl Default for InstanceShaderBindingOffsetAndFlags {
    fn default() -> Self {
        InstanceShaderBindingOffsetAndFlags::new(0, vk::GeometryInstanceFlagsKHR::empty())
    }
}

unsafe impl bytemuck::Zeroable for AccelerationStructureInstance {}
unsafe impl bytemuck::Pod for AccelerationStructureInstance {}

impl AccelerationStructureInstance {
    pub fn new(blas_address: DeviceAddress) -> Self {
        AccelerationStructureInstance {
            transform: Default::default(),
            custom_index_mask: Default::default(),
            shader_binding_offset_flags: Default::default(),
            acceleration_structure_reference: blas_address,
        }
    }

    pub fn with_transform(mut self, transform: TransformMatrix) -> Self {
        self.transform = transform;
        self
    }

    pub fn set_transform(&mut self, transform: TransformMatrix) -> &mut Self {
        self.transform = transform;
        self
    }
}
