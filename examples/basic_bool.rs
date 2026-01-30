//! Basic example showing a grid of chunks with a sphere spanning multiple chunks.
//! Uses a binary voxel field (Field<bool>) instead of SDF.
//!
//! Controls:
//! - Left click + drag: Rotate camera
//! - WASD: Move horizontally
//! - Space/Shift: Move up/down
//! - Ctrl: Speed boost

use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy_sculpter::prelude::*;
use chunky_bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SurfaceNetsPlugin)
        .register_sculptable::<BinaryField, bool>()
        .insert_resource(MeshSize(vec3(10., 10., 10.)))
        .add_systems(Startup, setup)
        .add_systems(Update, fly_camera)
        .run();
}

// ============================================================================
// Binary voxel field implementation
// ============================================================================

/// Binary voxel field - true = solid, false = air
#[derive(Component, Clone, Debug)]
pub struct BinaryField(pub Vec<bool>);

impl Default for BinaryField {
    fn default() -> Self {
        Self(vec![false; FIELD_VOLUME]) // All air
    }
}

impl Field<bool> for BinaryField {
    const SIZE: UVec3 = FIELD_SIZE;
    const DEFAULT: bool = false; // Outside = air

    fn data(&self) -> &[bool] {
        &self.0
    }

    fn data_mut(&mut self) -> &mut [bool] {
        &mut self.0
    }
}

impl Sculptable<bool> for BinaryField {
    fn to_iso(value: bool) -> f32 {
        if value { -1.0 } else { 1.0 } // true = inside (solid), false = outside (air)
    }
}

impl BinaryField {
    /// Fill a sphere with solid voxels.
    pub fn fill_sphere(&mut self, center: Vec3, radius: f32) {
        let radius_sq = radius * radius;
        for z in 0..FIELD_SIZE.z {
            for y in 0..FIELD_SIZE.y {
                for x in 0..FIELD_SIZE.x {
                    let pos = vec3(x as f32, y as f32, z as f32);
                    if pos.distance_squared(center) <= radius_sq {
                        self.set(x, y, z, true);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Camera
// ============================================================================

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
            speed: 20.0,
            sensitivity: 0.003,
            pitch: 0.0,
            yaw: 0.0,
        }
    }
}

// ============================================================================
// Setup
// ============================================================================

fn setup(mut commands: Commands, _mesh_size: Res<MeshSize>) {
    // Spawn a 3x3x3 grid of chunks with binary fields as children
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let chunk_pos = ivec3(x, y, z);

                // Create the chunk parent
                let chunk = commands.spawn((Chunk, ChunkPosition(chunk_pos))).id();

                // Create binary field for this chunk
                let mut field = BinaryField::default();

                // Create a sphere that spans multiple chunks
                let local_center = vec3(16.0, 16.0, 16.0);
                let global_offset = vec3(x as f32, y as f32, z as f32) * 32.0;
                let sphere_center = vec3(0.0, 0.0, 0.0);
                let local_sphere_center = sphere_center - global_offset + local_center;

                field.fill_sphere(local_sphere_center, 20.0);

                // Spawn field as child of chunk
                commands.entity(chunk).with_child((field, GenerateMesh));
            }
        }
    }

    // Camera with fly controls
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(30.0, 30.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam::default(),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn fly_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
) {
    let Ok((mut transform, mut fly_cam)) = query.single_mut() else {
        return;
    };

    // Mouse look (only when left mouse button is held)
    if mouse_buttons.pressed(MouseButton::Left) {
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

    // Speed boost with Ctrl
    let speed = if keyboard.pressed(KeyCode::ControlLeft) {
        fly_cam.speed * 3.0
    } else {
        fly_cam.speed
    };

    if velocity.length_squared() > 0.0 {
        velocity = velocity.normalize() * speed * time.delta_secs();
        transform.translation += velocity;
    }
}
