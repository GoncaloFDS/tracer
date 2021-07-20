use crate::material::Material;
use crate::render::mesh::Mesh;
use crate::render::renderer::Renderer;
use crate::Camera;
use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::utils::HashSet;
use bevy::window::{WindowCreated, WindowResized};
use bevy::winit::WinitWindows;

mod acceleration_structures;
mod buffer;
mod command_buffer;
mod debug;
mod descriptor;
mod device;
mod encoder;
mod framebuffer;
mod image;
mod instance;
pub mod mesh;
mod pass;
mod physical_device;
mod pipeline;
mod queue;
mod render_context;
mod render_pass;
pub mod renderer;
mod resources;
mod shader;
mod surface;
mod swapchain;
mod util;
pub mod vertex;

#[derive(Default)]
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_asset::<Mesh>()
            .add_asset::<Material>()
            .add_startup_system_to_stage(StartupStage::PreStartup, setup.system())
            .add_system(load_gltf_models.system())
            .add_system_to_stage(CoreStage::PreUpdate, window_resize.system())
            .add_system_to_stage(CoreStage::Update, draw.system())
            .add_system_to_stage(CoreStage::Last, world_cleanup.system());
    }
}

fn setup(
    mut commands: Commands,
    mut window_created_events: EventReader<WindowCreated>,
    winit_windows: Res<WinitWindows>,
) {
    let window_id = window_created_events
        .iter()
        .next()
        .map(|event| event.id)
        .unwrap();

    let winit_window = winit_windows.get_window(window_id).unwrap();
    let renderer = Renderer::new(winit_window);

    commands.insert_resource(renderer);
}

fn load_gltf_models(
    mut renderer: ResMut<Renderer>,
    meshes: Res<Assets<Mesh>>,
    mut mesh_events: EventReader<AssetEvent<Mesh>>,
) {
    let mut changed_meshes = HashSet::default();
    for event in mesh_events.iter() {
        match event {
            AssetEvent::Created { ref handle } => {
                tracing::info!("created mesh");
                changed_meshes.insert(handle.clone_weak());
            }
            AssetEvent::Modified { ref handle } => {
                tracing::info!("modified mesh");
                changed_meshes.insert(handle.clone_weak());
            }
            AssetEvent::Removed { ref handle } => {
                tracing::info!("removed mesh");
                changed_meshes.remove(handle);
            }
        }
    }

    for changed_mesh_handle in changed_meshes.iter() {
        if let Some(mesh) = meshes.get(changed_mesh_handle) {
            renderer.load_models(changed_mesh_handle, mesh);
        }
    }
}

fn draw(mut renderer: ResMut<Renderer>, mut query: Query<(&Camera, &GlobalTransform)>) {
    let (_camera, transform) = query.single_mut().unwrap();
    renderer.draw(transform);
}

fn window_resize(mut window_resized_event: EventReader<WindowResized>) {
    for event in window_resized_event.iter() {
        if event.width != 0.0 && event.height != 0.0 {
            tracing::debug!("window resized")
        }
    }
}

fn world_cleanup(mut commands: Commands, mut app_exit_events: EventReader<AppExit>) {
    if app_exit_events.iter().next().is_some() {
        commands.remove_resource::<Renderer>();
    }
}
