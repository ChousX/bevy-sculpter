//! Interactive 3D sculpting example with smooth brushes.
//!
//! Controls:
//! - Middle click + drag: Rotate camera
//! - Right click (hold): Smooth add material
//! - Middle click (hold): Smooth remove material  
//! - Shift + Right click: Hard add (CSG union)
//! - Shift + Left click: Hard remove (CSG subtract)
//! - B: Toggle blur/smooth brush
//! - Scroll wheel: Adjust brush size
//! - [ / ]: Adjust brush strength
//! - WASD/Space/Shift: Move camera

use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    window::PrimaryWindow,
};
use bevy_sculpter::prelude::*;
use chunky_bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ChunkyPlugin::default())
        .add_plugins(SurfaceNetsPlugin)
        .insert_resource(DensityFieldMeshSize(vec3(10., 10., 10.)))
        .init_resource::<SculptBrush>()
        .add_systems(Startup, (setup, show_chunks))
        .add_systems(
            Update,
            (fly_camera, sculpt_terrain, update_brush_preview, ui_text),
        )
        .run();
}

fn show_chunks(mut show_chunks: ResMut<NextState<ChunkBoundryVisualizer>>) {
    show_chunks.set(ChunkBoundryVisualizer::On);
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum BrushMode {
    #[default]
    Smooth,
    Hard,
    Blur,
}

#[derive(Resource)]
struct SculptBrush {
    radius: f32,
    min_radius: f32,
    max_radius: f32,
    strength: f32,
    min_strength: f32,
    max_strength: f32,
    falloff: f32,
    mode: BrushMode,
}

impl Default for SculptBrush {
    fn default() -> Self {
        Self {
            radius: 2.0,
            min_radius: 0.5,
            max_radius: 8.0,
            strength: 5.0, // Units per second for smooth brush
            min_strength: 0.5,
            max_strength: 20.0,
            falloff: 2.0, // Quadratic falloff
            mode: BrushMode::Smooth,
        }
    }
}

#[derive(Component)]
struct BrushPreview;

#[derive(Component)]
struct UiText;

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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let mut field = DensityField::new();
                let local_center = vec3(16.0, 16.0, 16.0);
                let global_offset = vec3(x as f32, y as f32, z as f32) * 32.0;
                let sphere_center = vec3(0.0, 0.0, 0.0);
                let local_sphere_center = sphere_center - global_offset + local_center;
                bevy_sculpter::helpers::fill_sphere(&mut field, local_sphere_center, 20.0);
                commands.spawn((Chunk, ChunkPos(ivec3(x, y, z)), field, DensityFieldDirty));
            }
        }
    }

    // Brush preview sphere
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.2, 0.8, 0.2, 0.3),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        })),
        Transform::from_scale(Vec3::ZERO),
        BrushPreview,
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(30.0, 30.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam::default(),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Text::new(""),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        UiText,
    ));
}

fn fly_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
    mut query: Query<(&mut Transform, &mut FlyCam)>,
    mut brush: ResMut<SculptBrush>,
) {
    let Ok((mut transform, mut fly_cam)) = query.single_mut() else {
        return;
    };

    // Toggle brush mode with B
    if keyboard.just_pressed(KeyCode::KeyB) {
        brush.mode = match brush.mode {
            BrushMode::Smooth => BrushMode::Blur,
            BrushMode::Blur => BrushMode::Smooth,
            BrushMode::Hard => BrushMode::Smooth,
        };
    }

    // Adjust strength with [ and ]
    if keyboard.just_pressed(KeyCode::BracketLeft) {
        brush.strength = (brush.strength - 1.0).max(brush.min_strength);
    }
    if keyboard.just_pressed(KeyCode::BracketRight) {
        brush.strength = (brush.strength + 1.0).min(brush.max_strength);
    }

    if mouse_buttons.pressed(MouseButton::Middle) {
        for motion in mouse_motion.read() {
            fly_cam.yaw -= motion.delta.x * fly_cam.sensitivity;
            fly_cam.pitch -= motion.delta.y * fly_cam.sensitivity;
            fly_cam.pitch = fly_cam.pitch.clamp(-1.5, 1.5);
        }
        transform.rotation = Quat::from_euler(EulerRot::YXZ, fly_cam.yaw, fly_cam.pitch, 0.0);
    } else {
        mouse_motion.clear();
    }

    for ev in scroll.read() {
        brush.radius = (brush.radius + ev.y * 0.2).clamp(brush.min_radius, brush.max_radius);
    }

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

fn sculpt_terrain(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    window_q: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    mut chunks: Query<(&ChunkPos, &mut DensityField)>,
    mesh_size: Res<DensityFieldMeshSize>,
    brush: Res<SculptBrush>,
    mut commands: Commands,
    chunk_entities: Query<Entity, With<ChunkPos>>,
) {
    let adding = mouse_buttons.pressed(MouseButton::Right);
    let removing = mouse_buttons.pressed(MouseButton::Left);

    if !adding && !removing {
        return;
    }

    let Ok(window) = window_q.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else {
        return;
    };

    let Some(hit_point) = raycast_terrain(&chunks, &mesh_size, ray) else {
        return;
    };

    let world_brush_radius = brush.radius;
    let chunk_world_size = mesh_size.0;
    let use_hard_brush =
        keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);

    for (chunk_pos, mut field) in chunks.iter_mut() {
        let chunk_world_origin = chunk_pos.0.as_vec3() * chunk_world_size;
        let local_hit = hit_point - chunk_world_origin;

        let scale = Vec3::new(32.0, 32.0, 32.0) / chunk_world_size;
        let grid_center = local_hit * scale;
        let grid_radius = world_brush_radius * scale.x;

        // AABB check
        let chunk_min = Vec3::ZERO;
        let chunk_max = Vec3::splat(32.0);
        let brush_min = grid_center - Vec3::splat(grid_radius);
        let brush_max = grid_center + Vec3::splat(grid_radius);

        if brush_max.x < chunk_min.x
            || brush_min.x > chunk_max.x
            || brush_max.y < chunk_min.y
            || brush_min.y > chunk_max.y
            || brush_max.z < chunk_min.z
            || brush_min.z > chunk_max.z
        {
            continue;
        }

        if use_hard_brush {
            // Hard CSG brush (instant)
            bevy_sculpter::helpers::brush_sphere(&mut field, grid_center, grid_radius, adding);
        } else {
            match brush.mode {
                BrushMode::Smooth => {
                    // Smooth brush: rate is strength per second
                    // Negative rate = add material (decrease SDF)
                    // Positive rate = remove material (increase SDF)
                    let rate = if adding {
                        -brush.strength
                    } else {
                        brush.strength
                    };
                    bevy_sculpter::helpers::brush_smooth_timed(
                        &mut field,
                        grid_center,
                        grid_radius,
                        rate,
                        time.delta_secs(),
                        brush.falloff,
                    );
                }
                BrushMode::Blur => {
                    // Blur/smooth brush
                    bevy_sculpter::helpers::brush_blur(
                        &mut field,
                        grid_center,
                        grid_radius,
                        brush.strength * 0.1 * time.delta_secs(),
                        brush.falloff,
                    );
                }
                BrushMode::Hard => {
                    bevy_sculpter::helpers::brush_sphere(
                        &mut field,
                        grid_center,
                        grid_radius,
                        adding,
                    );
                }
            }
        }
    }

    for entity in chunk_entities.iter() {
        commands.entity(entity).insert(DensityFieldDirty);
    }
}

fn raycast_terrain(
    chunks: &Query<(&ChunkPos, &mut DensityField)>,
    mesh_size: &DensityFieldMeshSize,
    ray: Ray3d,
) -> Option<Vec3> {
    let chunk_world_size = mesh_size.0;
    let max_dist = 200.0;
    let step = 0.1;
    let mut t = 0.0;

    while t < max_dist {
        let point = ray.origin + ray.direction * t;
        let chunk_coord = (point / chunk_world_size).floor().as_ivec3();

        for (chunk_pos, field) in chunks.iter() {
            if chunk_pos.0 != chunk_coord {
                continue;
            }

            let chunk_origin = chunk_pos.0.as_vec3() * chunk_world_size;
            let local_pos = point - chunk_origin;
            let scale = Vec3::new(32.0, 32.0, 32.0) / chunk_world_size;
            let grid_pos = local_pos * scale;

            if grid_pos.x >= 0.0
                && grid_pos.x < 32.0
                && grid_pos.y >= 0.0
                && grid_pos.y < 32.0
                && grid_pos.z >= 0.0
                && grid_pos.z < 32.0
            {
                let density = field.get(grid_pos.x as u32, grid_pos.y as u32, grid_pos.z as u32);
                if density < 0.0 {
                    return Some(point);
                }
            }
        }
        t += step;
    }
    None
}

fn update_brush_preview(
    window_q: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<FlyCam>>,
    chunks: Query<(&ChunkPos, &mut DensityField)>,
    mesh_size: Res<DensityFieldMeshSize>,
    brush: Res<SculptBrush>,
    mut preview_q: Query<(&mut Transform, &MeshMaterial3d<StandardMaterial>), With<BrushPreview>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Ok((mut preview_transform, mat_handle)) = preview_q.single_mut() else {
        return;
    };
    let Ok(window) = window_q.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        preview_transform.scale = Vec3::ZERO;
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else {
        return;
    };

    if let Some(hit) = raycast_terrain(&chunks, &mesh_size, ray) {
        preview_transform.translation = hit;
        preview_transform.scale = Vec3::splat(brush.radius);

        // Change color based on mode
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.base_color = match brush.mode {
                BrushMode::Smooth => Color::srgba(0.2, 0.8, 0.2, 0.3),
                BrushMode::Blur => Color::srgba(0.2, 0.2, 0.8, 0.3),
                BrushMode::Hard => Color::srgba(0.8, 0.2, 0.2, 0.3),
            };
        }
    } else {
        preview_transform.scale = Vec3::ZERO;
    }
}

fn ui_text(brush: Res<SculptBrush>, mut text_q: Query<&mut Text, With<UiText>>) {
    let Ok(mut text) = text_q.single_mut() else {
        return;
    };

    let mode_str = match brush.mode {
        BrushMode::Smooth => "Smooth (continuous)",
        BrushMode::Blur => "Blur/Smooth surface",
        BrushMode::Hard => "Hard (CSG)",
    };

    *text = Text::new(format!(
        "Sculpt Controls:\n\
         Middle Click + Drag: Rotate camera\n\
         Right Click (hold): Add material\n\
         Left Click (hold): Remove material\n\
         Ctrl + Click: Hard brush (instant CSG)\n\
         \n\
         B: Toggle brush mode\n\
         Scroll: Brush size ({:.1})\n\
         [ ]: Brush strength ({:.1})\n\
         \n\
         Mode: {}",
        brush.radius, brush.strength, mode_str
    ));
}
