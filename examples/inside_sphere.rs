// examples/inside_sphere.rs
//! Camera positioned inside a hollow sphere to view the interior surface.
//!
//! Controls:
//! - Mouse: Look around (captured by default)
//! - WASD: Move horizontally
//! - Space/Shift: Move up/down
//! - Esc: Release/capture mouse cursor

use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use bevy_sculpter::prelude::*;
use chunky_bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SurfaceNetsPlugin)
        //may want put this in the field type
        .insert_resource(DensityFieldMeshSize(vec3(10., 10., 10.)))
        .add_systems(Startup, setup)
        .add_systems(Update, (fly_camera, toggle_cursor))
        .run();
}

#[derive(Component)]
struct FlyCam {
    speed: f32,
    sensitivity: f32,
    pitch: f32,
    yaw: f32,
}

impl Default for FlyCam {
    fn default() -> Self {
        Self {
            speed: 5.0,
            sensitivity: 0.003,
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

fn setup(
    mut commands: Commands,
    mut windows: Query<&mut CursorOptions, With<PrimaryWindow>>,
    cmr: Res<DefaultChunkManager>,
) {
    // Create a hollow sphere by subtracting a smaller sphere from a larger one
    let mut field = DefaultIsoField::new();
    let center = vec3(16.0, 16.0, 16.0);
    let outer_radius = 15.0;
    let inner_radius = 12.0;

    // Fill with outer sphere
    bevy_sculpter::helpers::fill_sphere(&mut field, center, outer_radius);

    // Carve out inner sphere to make it hollow
    bevy_sculpter::helpers::brush_sphere(&mut field, center, inner_radius, false);

    commands.spawn((
        Chunk(cmr.entity),
        ChunkPositon(ivec3(0, 0, 0)),
        field,
        GenerateMesh,
    ));

    // Camera starts at the center, inside the hollow sphere
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(5.0, 5.0, 5.0).looking_at(Vec3::new(5.0, 5.0, 6.0), Vec3::Y),
        FlyCam::default(),
    ));

    // Ambient light to see the interior
    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: 800.0,
        affects_lightmapped_meshes: false,
    });

    // Directional light for some depth
    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::new(5.0, 5.0, 5.0), Vec3::Y),
    ));

    // Grab cursor on startup
    if let Ok(mut window) = windows.single_mut() {
        window.grab_mode = CursorGrabMode::Locked;
        window.visible = false;
    }
}

fn toggle_cursor(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        if let Ok(mut cursor) = cursor.single_mut() {
            match cursor.grab_mode {
                CursorGrabMode::Locked => {
                    cursor.grab_mode = CursorGrabMode::None;
                    cursor.visible = true;
                }
                _ => {
                    cursor.grab_mode = CursorGrabMode::Locked;
                    cursor.visible = false;
                }
            }
        }
    }
}

fn fly_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
) {
    let Ok((mut transform, mut fly_cam)) = query.single_mut() else {
        return;
    };

    let cursor = windows.single().ok();
    let cursor_locked = cursor.map_or(false, |c| matches!(c.grab_mode, CursorGrabMode::Locked));

    // Mouse look (only when cursor is locked)
    if cursor_locked {
        for motion in mouse_motion.read() {
            fly_cam.yaw -= motion.delta.x * fly_cam.sensitivity;
            fly_cam.pitch -= motion.delta.y * fly_cam.sensitivity;
            fly_cam.pitch = fly_cam.pitch.clamp(-1.5, 1.5);
        }

        transform.rotation = Quat::from_euler(EulerRot::YXZ, fly_cam.yaw, fly_cam.pitch, 0.0);
    } else {
        mouse_motion.clear();
    }

    // Keyboard movement
    let mut velocity = Vec3::ZERO;
    let forward = transform.forward();
    let right = transform.right();

    if keyboard.pressed(KeyCode::KeyW) {
        velocity += *forward;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        velocity -= *forward;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        velocity -= *right;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        velocity += *right;
    }
    if keyboard.pressed(KeyCode::Space) {
        velocity += Vec3::Y;
    }
    if keyboard.pressed(KeyCode::ShiftLeft) {
        velocity -= Vec3::Y;
    }

    if velocity.length_squared() > 0.0 {
        velocity = velocity.normalize() * fly_cam.speed * time.delta_secs();
        transform.translation += velocity;
    }
}
