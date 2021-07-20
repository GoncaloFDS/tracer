use bevy::math::vec3;
use bevy::prelude::*;

use crate::camera_controller::{CameraController, CameraPlugin};
use crate::gltf::GltfPlugin;
use crate::render::RenderPlugin;

mod camera_controller;
mod gltf;
mod material;
mod render;

fn main() {
    App::build()
        .insert_resource(bevy::log::LogSettings {
            level: bevy::utils::tracing::Level::DEBUG,
            ..Default::default()
        })
        .insert_resource(bevy::window::WindowDescriptor {
            width: 800.0,
            height: 600.0,
            title: "RDX".to_string(),
            ..Default::default()
        })
        .add_plugin(bevy::log::LogPlugin::default())
        .add_plugin(bevy::core::CorePlugin::default())
        .add_plugin(bevy::transform::TransformPlugin::default())
        .add_plugin(bevy::diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(bevy::input::InputPlugin::default())
        .add_plugin(bevy::window::WindowPlugin::default())
        .add_plugin(bevy::winit::WinitPlugin::default())
        .add_plugin(bevy::asset::AssetPlugin::default())
        .add_plugin(bevy::scene::ScenePlugin::default())
        .add_plugin(GltfPlugin::default())
        .add_plugin(CameraPlugin::default())
        .add_plugin(RenderPlugin::default())
        .add_startup_system(setup.system())
        .run()
}

#[derive(Default)]
pub struct Camera;

#[derive(Bundle, Default)]
pub struct CameraBundle {
    pub global_transform: GlobalTransform,
    pub transform: Transform,
    pub camera: Camera,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let mut camera = CameraBundle::default();
    camera.transform.translation = vec3(0.0, 0.0, 1.0);
    camera.transform.looking_at(Vec3::ZERO, Vec3::Y);
    commands
        .spawn()
        .insert_bundle(camera)
        .insert(CameraController::default());

    commands.spawn_scene(asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"));
}
