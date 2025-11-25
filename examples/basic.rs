// examples/terrain.rs
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
                let local_center = vec3(16.0, 16.0, 16.0); // Center of chunk
                let global_offset = vec3(x as f32, y as f32, z as f32) * 32.0;
                let sphere_center = vec3(0.0, 0.0, 0.0); // World center
                let local_sphere_center = sphere_center - global_offset + local_center;

                field.fill_sphere(local_sphere_center, 20.0);

                commands.spawn((Chunk, ChunkPos(ivec3(x, y, z)), field, DensityFieldDirty));
            }
        }
    }

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(30.0, 30.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Light
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
