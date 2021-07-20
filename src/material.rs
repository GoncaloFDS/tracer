use bevy::reflect::TypeUuid;
use glam::Vec4;

#[derive(Debug, TypeUuid)]
#[uuid = "dace545e-4bc6-4595-a79d-c224fc694975"]
pub struct Material {
    pub base_color: Vec4,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            base_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
        }
    }
}
