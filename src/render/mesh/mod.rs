mod conversions;

use crate::material::Material;
use crate::render::acceleration_structures::IndexData;
use crate::render::pipeline::vertex_format::VertexFormat;
use crate::render::{
    acceleration_structures::{
        AccelerationStructureBuildGeometryInfo, AccelerationStructureGeometry,
        AccelerationStructureGeometryInfo, AccelerationStructureInfo, AccelerationStructureLevel,
    },
    buffer::{BufferInfo, BufferRegion},
    device::Device,
    encoder::Encoder,
    resources::{AccelerationStructure, Buffer},
    vertex::{Indices, PrimitiveTopology},
};
use bevy::asset::Handle;
use bevy::ecs::bundle::Bundle;
use bevy::reflect::TypeUuid;
use bumpalo::Bump;
use bytemuck::cast_slice;
use erupt::vk;
use glam::Vec3;
use std::borrow::Cow;
use std::collections::BTreeMap;

#[derive(Bundle)]
pub struct MeshBundle {
    pub mesh: Handle<Mesh>,
    pub material: Handle<Material>,
}

#[derive(Debug, TypeUuid, Clone)]
#[uuid = "8ecbac0f-f545-4473-ad43-e1f4243af51e"]
pub struct Mesh {
    primitive_topology: PrimitiveTopology,
    attributes: BTreeMap<Cow<'static, str>, VertexAttributeValues>,
    indices: Option<Indices>,
}

impl Mesh {
    pub const ATTRIBUTE_COLOR: &'static str = "Vertex_Color";
    pub const ATTRIBUTE_NORMAL: &'static str = "Vertex_Normal";
    pub const ATTRIBUTE_TANGENT: &'static str = "Vertex_Tangent";
    pub const ATTRIBUTE_POSITION: &'static str = "Vertex_Position";
    pub const ATTRIBUTE_UV_0: &'static str = "Vertex_Uv";
    pub const ATTRIBUTE_JOINT_WEIGHT: &'static str = "Vertex_JointWeight";
    pub const ATTRIBUTE_JOINT_INDEX: &'static str = "Vertex_JointIndex";

    pub fn new(primitive_topology: PrimitiveTopology) -> Self {
        Mesh {
            primitive_topology,
            attributes: Default::default(),
            indices: None,
        }
    }

    pub fn primitive_topology(&self) -> PrimitiveTopology {
        self.primitive_topology
    }

    pub fn set_attribute(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        values: impl Into<VertexAttributeValues>,
    ) {
        let values: VertexAttributeValues = values.into();
        self.attributes.insert(name.into(), values);
    }

    pub fn attribute(&self, name: impl Into<Cow<'static, str>>) -> Option<&VertexAttributeValues> {
        self.attributes.get(&name.into())
    }

    pub fn attribute_mut(
        &mut self,
        name: impl Into<Cow<'static, str>>,
    ) -> Option<&mut VertexAttributeValues> {
        self.attributes.get_mut(&name.into())
    }

    pub fn set_indices(&mut self, indices: Option<Indices>) {
        self.indices = indices;
    }

    pub fn indices(&self) -> Option<&Indices> {
        self.indices.as_ref()
    }

    pub fn indices_mut(&mut self) -> Option<&mut Indices> {
        self.indices.as_mut()
    }

    pub fn get_index_buffer_bytes(&self) -> Option<&[u8]> {
        self.indices.as_ref().map(|indices| match &indices {
            Indices::U16(indices) => cast_slice(&indices[..]),
            Indices::U32(indices) => cast_slice(&indices[..]),
        })
    }

    pub fn count_vertices(&self) -> usize {
        let mut vertex_count: Option<usize> = None;
        for (attribute_name, attribute_data) in self.attributes.iter() {
            let attribute_len = attribute_data.len();
            if let Some(previous_vertex_count) = vertex_count {
                assert_eq!(previous_vertex_count, attribute_len,
                           "Attribute {} has a different vertex count ({}) than other attributes ({}) in this mesh.", attribute_name, attribute_len, previous_vertex_count);
            }
            vertex_count = Some(attribute_len);
        }

        vertex_count.unwrap_or(0)
    }

    /// Duplicates the vertex attributes so that no vertices are shared.
    ///
    /// This can dramatically increase the vertex count, so make sure this is what you want.
    /// Does nothing if no [Indices] are set.
    pub fn duplicate_vertices(&mut self) {
        fn duplicate<T: Copy>(values: &[T], indices: impl Iterator<Item = usize>) -> Vec<T> {
            indices.map(|i| values[i]).collect()
        }

        assert!(
            matches!(self.primitive_topology, PrimitiveTopology::TriangleList),
            "can only duplicate vertices for `TriangleList`s"
        );

        let indices = match self.indices.take() {
            Some(indices) => indices,
            None => return,
        };
        for (_, attributes) in self.attributes.iter_mut() {
            let indices = indices.iter();
            match attributes {
                VertexAttributeValues::Float32(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Sint32(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Uint32(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Float32x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Sint32x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Uint32x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Float32x3(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Sint32x3(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Uint32x3(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Sint32x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Uint32x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Float32x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Sint16x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Snorm16x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Uint16x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Unorm16x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Sint16x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Snorm16x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Uint16x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Unorm16x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Sint8x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Snorm8x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Uint8x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Unorm8x2(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Sint8x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Snorm8x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Uint8x4(vec) => *vec = duplicate(&vec, indices),
                VertexAttributeValues::Unorm8x4(vec) => *vec = duplicate(&vec, indices),
            }
        }
    }

    /// Calculates the [`Mesh::ATTRIBUTE_NORMAL`] of a mesh.
    ///
    /// Panics if [`Indices`] are set.
    /// Consider calling [Mesh::duplicate_vertices] or export your mesh with normal attributes.
    pub fn compute_flat_normals(&mut self) {
        if self.indices().is_some() {
            panic!("`compute_flat_normals` can't work on indexed geometry. Consider calling `Mesh::duplicate_vertices`.");
        }

        let positions = self
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .expect("`Mesh::ATTRIBUTE_POSITION` vertex attributes should be of type `float3`");

        let normals: Vec<_> = positions
            .chunks_exact(3)
            .map(|p| face_normal(p[0], p[1], p[2]))
            .flat_map(|normal| std::array::IntoIter::new([normal, normal, normal]))
            .collect();

        self.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    }

    pub fn build_triangle_blas<'a>(
        &self,
        device: &Device,
        encoder: &mut Encoder<'a>,
        bump: &'a Bump,
    ) -> (AccelerationStructure, Buffer, Buffer, Buffer) {
        let vertices = self.attributes.get(Mesh::ATTRIBUTE_POSITION).unwrap();
        let vertex_count = vertices.len() as u64;
        let vertex_stride = VertexFormat::from(vertices).get_size();
        let vertex_buffer = device.create_buffer_with_data(
            BufferInfo {
                align: 255,
                size: vertex_stride * vertex_count,
                usage_flags: vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS
                    | gpu_alloc::UsageFlags::HOST_ACCESS,
            },
            vertices.get_bytes(),
        );

        let indices = self.indices().expect("Mesh without indices");
        let triangle_count = indices.len() / 3;

        let index_buffer = device.create_buffer_with_data(
            BufferInfo {
                align: 255,
                size: indices.get_total_size() as u64,
                usage_flags: vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS
                    | gpu_alloc::UsageFlags::HOST_ACCESS,
            },
            self.get_index_buffer_bytes().unwrap(),
        );

        let sizes = device.get_acceleration_structure_build_sizes(
            AccelerationStructureLevel::Bottom,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR,
            &[AccelerationStructureGeometryInfo::Triangles {
                max_primitive_count: triangle_count as u32,
                max_vertex_count: vertex_count as u32,
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
            usage_flags: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
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
            index_data: match indices {
                Indices::U16(_) => Some(IndexData::U16(index_buffer.device_address().unwrap())),
                Indices::U32(_) => Some(IndexData::U32(index_buffer.device_address().unwrap())),
            },
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
}

#[derive(Clone, Debug)]
pub enum VertexAttributeValues {
    Float32(Vec<f32>),
    Sint32(Vec<i32>),
    Uint32(Vec<u32>),
    Float32x2(Vec<[f32; 2]>),
    Sint32x2(Vec<[i32; 2]>),
    Uint32x2(Vec<[u32; 2]>),
    Float32x3(Vec<[f32; 3]>),
    Sint32x3(Vec<[i32; 3]>),
    Uint32x3(Vec<[u32; 3]>),
    Float32x4(Vec<[f32; 4]>),
    Sint32x4(Vec<[i32; 4]>),
    Uint32x4(Vec<[u32; 4]>),
    Sint16x2(Vec<[i16; 2]>),
    Snorm16x2(Vec<[i16; 2]>),
    Uint16x2(Vec<[u16; 2]>),
    Unorm16x2(Vec<[u16; 2]>),
    Sint16x4(Vec<[i16; 4]>),
    Snorm16x4(Vec<[i16; 4]>),
    Uint16x4(Vec<[u16; 4]>),
    Unorm16x4(Vec<[u16; 4]>),
    Sint8x2(Vec<[i8; 2]>),
    Snorm8x2(Vec<[i8; 2]>),
    Uint8x2(Vec<[u8; 2]>),
    Unorm8x2(Vec<[u8; 2]>),
    Sint8x4(Vec<[i8; 4]>),
    Snorm8x4(Vec<[i8; 4]>),
    Uint8x4(Vec<[u8; 4]>),
    Unorm8x4(Vec<[u8; 4]>),
}

impl VertexAttributeValues {
    pub fn len(&self) -> usize {
        match *self {
            VertexAttributeValues::Float32(ref values) => values.len(),
            VertexAttributeValues::Sint32(ref values) => values.len(),
            VertexAttributeValues::Uint32(ref values) => values.len(),
            VertexAttributeValues::Float32x2(ref values) => values.len(),
            VertexAttributeValues::Sint32x2(ref values) => values.len(),
            VertexAttributeValues::Uint32x2(ref values) => values.len(),
            VertexAttributeValues::Float32x3(ref values) => values.len(),
            VertexAttributeValues::Sint32x3(ref values) => values.len(),
            VertexAttributeValues::Uint32x3(ref values) => values.len(),
            VertexAttributeValues::Float32x4(ref values) => values.len(),
            VertexAttributeValues::Sint32x4(ref values) => values.len(),
            VertexAttributeValues::Uint32x4(ref values) => values.len(),
            VertexAttributeValues::Sint16x2(ref values) => values.len(),
            VertexAttributeValues::Snorm16x2(ref values) => values.len(),
            VertexAttributeValues::Uint16x2(ref values) => values.len(),
            VertexAttributeValues::Unorm16x2(ref values) => values.len(),
            VertexAttributeValues::Sint16x4(ref values) => values.len(),
            VertexAttributeValues::Snorm16x4(ref values) => values.len(),
            VertexAttributeValues::Uint16x4(ref values) => values.len(),
            VertexAttributeValues::Unorm16x4(ref values) => values.len(),
            VertexAttributeValues::Sint8x2(ref values) => values.len(),
            VertexAttributeValues::Snorm8x2(ref values) => values.len(),
            VertexAttributeValues::Uint8x2(ref values) => values.len(),
            VertexAttributeValues::Unorm8x2(ref values) => values.len(),
            VertexAttributeValues::Sint8x4(ref values) => values.len(),
            VertexAttributeValues::Snorm8x4(ref values) => values.len(),
            VertexAttributeValues::Uint8x4(ref values) => values.len(),
            VertexAttributeValues::Unorm8x4(ref values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn as_float3(&self) -> Option<&[[f32; 3]]> {
        match self {
            VertexAttributeValues::Float32x3(values) => Some(values),
            _ => None,
        }
    }

    pub fn get_bytes(&self) -> &[u8] {
        match self {
            VertexAttributeValues::Float32(values) => cast_slice(&values[..]),
            VertexAttributeValues::Sint32(values) => cast_slice(&values[..]),
            VertexAttributeValues::Uint32(values) => cast_slice(&values[..]),
            VertexAttributeValues::Float32x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Sint32x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Uint32x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Float32x3(values) => cast_slice(&values[..]),
            VertexAttributeValues::Sint32x3(values) => cast_slice(&values[..]),
            VertexAttributeValues::Uint32x3(values) => cast_slice(&values[..]),
            VertexAttributeValues::Float32x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Sint32x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Uint32x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Sint16x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Snorm16x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Uint16x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Unorm16x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Sint16x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Snorm16x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Uint16x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Unorm16x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Sint8x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Snorm8x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Uint8x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Unorm8x2(values) => cast_slice(&values[..]),
            VertexAttributeValues::Sint8x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Snorm8x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Uint8x4(values) => cast_slice(&values[..]),
            VertexAttributeValues::Unorm8x4(values) => cast_slice(&values[..]),
        }
    }
}

impl From<&VertexAttributeValues> for VertexFormat {
    fn from(values: &VertexAttributeValues) -> Self {
        match values {
            VertexAttributeValues::Float32(_) => VertexFormat::Float32,
            VertexAttributeValues::Sint32(_) => VertexFormat::Sint32,
            VertexAttributeValues::Uint32(_) => VertexFormat::Uint32,
            VertexAttributeValues::Float32x2(_) => VertexFormat::Float32x2,
            VertexAttributeValues::Sint32x2(_) => VertexFormat::Sint32x2,
            VertexAttributeValues::Uint32x2(_) => VertexFormat::Uint32x2,
            VertexAttributeValues::Float32x3(_) => VertexFormat::Float32x3,
            VertexAttributeValues::Sint32x3(_) => VertexFormat::Sint32x3,
            VertexAttributeValues::Uint32x3(_) => VertexFormat::Uint32x3,
            VertexAttributeValues::Float32x4(_) => VertexFormat::Float32x4,
            VertexAttributeValues::Sint32x4(_) => VertexFormat::Sint32x4,
            VertexAttributeValues::Uint32x4(_) => VertexFormat::Uint32x4,
            VertexAttributeValues::Sint16x2(_) => VertexFormat::Sint16x2,
            VertexAttributeValues::Snorm16x2(_) => VertexFormat::Snorm16x2,
            VertexAttributeValues::Uint16x2(_) => VertexFormat::Uint16x2,
            VertexAttributeValues::Unorm16x2(_) => VertexFormat::Unorm16x2,
            VertexAttributeValues::Sint16x4(_) => VertexFormat::Sint16x4,
            VertexAttributeValues::Snorm16x4(_) => VertexFormat::Snorm16x4,
            VertexAttributeValues::Uint16x4(_) => VertexFormat::Uint16x4,
            VertexAttributeValues::Unorm16x4(_) => VertexFormat::Unorm16x4,
            VertexAttributeValues::Sint8x2(_) => VertexFormat::Sint8x2,
            VertexAttributeValues::Snorm8x2(_) => VertexFormat::Snorm8x2,
            VertexAttributeValues::Uint8x2(_) => VertexFormat::Uint8x2,
            VertexAttributeValues::Unorm8x2(_) => VertexFormat::Unorm8x2,
            VertexAttributeValues::Sint8x4(_) => VertexFormat::Sint8x4,
            VertexAttributeValues::Snorm8x4(_) => VertexFormat::Snorm8x4,
            VertexAttributeValues::Uint8x4(_) => VertexFormat::Uint8x4,
            VertexAttributeValues::Unorm8x4(_) => VertexFormat::Unorm8x4,
        }
    }
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let (a, b, c) = (Vec3::from(a), Vec3::from(b), Vec3::from(c));
    (b - a).cross(c - a).normalize().into()
}
