// examples/basic.rs
use bevy::prelude::*;
use chunky::prelude::*;
use sculpter::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ChunkyPlugin::default())
        .add_plugins(SurfaceNetsPlugin)
        .insert_resource(DensityFieldMeshSize(vec3(10., 10., 10.)))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    // Spawn a 3x3x3 grid of chunks with density fields
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let mut field = DensityField::new();

                // Create a sphere that spans multiple chunks
                // The sphere is centered at world origin
                // Each chunk needs to know where the sphere surface is relative to its local grid
                let chunk_world_offset = vec3(x as f32, y as f32, z as f32) * 32.0; // Grid units
                let sphere_center_world = vec3(0.0, 0.0, 0.0); // World center in grid units
                let local_sphere_center =
                    sphere_center_world - chunk_world_offset + vec3(16.0, 16.0, 16.0);

                field.fill_sphere(local_sphere_center, 20.0);

                commands.spawn((Chunk, ChunkPos(ivec3(x, y, z)), field, DensityFieldDirty));
            }
        }
    }

    // Camera - position it to see the sphere
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(25.0, 25.0, 25.0).looking_at(Vec3::ZERO, Vec3::Y),
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
