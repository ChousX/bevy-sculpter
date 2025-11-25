// Example: How to use the Surface Nets plugin
use bevy::prelude::*;
use sculpter::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, SculpterPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut dimensions: ResMut<DensityFieldSize>,
    mut mesh_size: ResMut<DensityFieldMeshSize>,
) {
    // STEP 1: Configure grid dimensions
    // This determines the resolution of your surface
    *dimensions = DensityFieldSize(UVec3::new(6, 6, 6));

    // STEP 2: Configure physical mesh size in world space
    *mesh_size = DensityFieldMeshSize(Vec3::splat(10.0));

    // STEP 3: Generate a density field
    // For this example, we'll create a sphere using a signed distance function (SDF)
    let density_field = generate_sphere_sdf(*dimensions, 0.3);

    // STEP 4: Spawn an entity with the DensityField component
    // The plugin will automatically:
    // - Create GPU buffers
    // - Run compute shaders
    // - Read back results
    // - Build and attach a mesh
    commands.spawn((
        DensityField(density_field),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // STEP 5: Spawn a camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(15.0, 15.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // STEP 6: Add a light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, -0.5, 0.0)),
    ));
}

// ============================================
// Helper Functions to Generate Density Fields
// ============================================

/// Generate a sphere using signed distance function
fn generate_sphere_sdf(dimensions: DensityFieldSize, radius_ratio: f32) -> Vec<f32> {
    let dims = dimensions.0;
    let center = dims.as_vec3() * 0.5;
    let radius = dims.x.min(dims.y).min(dims.z) as f32 * radius_ratio;

    let mut field = Vec::with_capacity(dimensions.density_count() as usize);

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let pos = Vec3::new(x as f32, y as f32, z as f32);
                let distance = pos.distance(center) - radius;
                field.push(distance);
            }
        }
    }

    field
}

/// Generate a torus using signed distance function
fn generate_torus_sdf(
    dimensions: DensityFieldSize,
    major_radius: f32,
    minor_radius: f32,
) -> Vec<f32> {
    let dims = dimensions.0;
    let center = dims.as_vec3() * 0.5;

    let mut field = Vec::with_capacity(dimensions.density_count() as usize);

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let pos = Vec3::new(x as f32, y as f32, z as f32) - center;

                // Torus SDF formula
                let xz_dist = Vec2::new(pos.x, pos.z).length() - major_radius;
                let distance = Vec2::new(xz_dist, pos.y).length() - minor_radius;

                field.push(distance);
            }
        }
    }

    field
}

/// Generate a box using signed distance function
fn generate_box_sdf(dimensions: DensityFieldSize, half_extents: Vec3) -> Vec<f32> {
    let dims = dimensions.0;
    let center = dims.as_vec3() * 0.5;

    let mut field = Vec::with_capacity(dimensions.density_count() as usize);

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let pos = Vec3::new(x as f32, y as f32, z as f32) - center;

                // Box SDF formula
                let q = pos.abs() - half_extents;
                let distance = q.max(Vec3::ZERO).length() + q.max_element().min(0.0);

                field.push(distance);
            }
        }
    }

    field
}

/// Generate a custom field by combining multiple SDFs
fn generate_combined_sdf(dimensions: DensityFieldSize) -> Vec<f32> {
    let dims = dimensions.0;
    let center = dims.as_vec3() * 0.5;

    let mut field = Vec::with_capacity(dimensions.density_count() as usize);

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let pos = Vec3::new(x as f32, y as f32, z as f32);

                // Sphere 1
                let sphere1_dist = (pos - center).length() - 8.0;

                // Sphere 2 (offset)
                let sphere2_center = center + Vec3::new(6.0, 0.0, 0.0);
                let sphere2_dist = (pos - sphere2_center).length() - 8.0;

                // Combine using smooth minimum (creates a smooth blend)
                let k = 2.0; // Smoothness factor
                let distance = smooth_min(sphere1_dist, sphere2_dist, k);

                field.push(distance);
            }
        }
    }

    field
}

/// Smooth minimum function for blending SDFs
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b.lerp(a, h) - k * h * (1.0 - h)
}

// ============================================
// Advanced Example: Multiple Meshes
// ============================================

fn spawn_multiple_meshes(mut commands: Commands, dimensions: Res<DensityFieldSize>) {
    // Spawn multiple entities with different SDFs

    // Sphere at position 1
    commands.spawn((
        DensityField(generate_sphere_sdf(*dimensions, 0.4)),
        Transform::from_xyz(-15.0, 0.0, 0.0),
    ));

    // Torus at position 2
    commands.spawn((
        DensityField(generate_torus_sdf(*dimensions, 10.0, 4.0)),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Box at position 3
    commands.spawn((
        DensityField(generate_box_sdf(*dimensions, Vec3::new(8.0, 8.0, 8.0))),
        Transform::from_xyz(15.0, 0.0, 0.0),
    ));
}

// ============================================
// Advanced Example: Dynamic Updates
// ============================================

#[derive(Component)]
struct AnimatedSDF {
    time: f32,
}

fn update_animated_sdf(
    time: Res<Time>,
    dimensions: Res<DensityFieldSize>,
    mut query: Query<(&mut DensityField, &mut AnimatedSDF)>,
) {
    for (mut density_field, mut animated) in &mut query {
        animated.time += time.delta_secs();

        // Regenerate the density field with animated parameters
        let radius = 0.3 + (animated.time * 2.0).sin() * 0.1;
        density_field.0 = generate_sphere_sdf(*dimensions, radius);

        // Note: You'll need to re-trigger the compute pipeline
        // This might require adding a "dirty" flag system
    }
}

// ============================================
// Debugging Tips
// ============================================

// 1. Start with small grids (8x8x8) to verify correctness
// 2. Use simple shapes (sphere) first
// 3. Check logs for vertex/face counts
// 4. Visualize intermediate results if needed
// 5. Increase grid resolution gradually (16x16x16, then 32x32x32, etc.)

// Common issues:
// - No mesh appears: Check if density field has sign changes
// - Mesh looks wrong: Verify SDF is correct (negative inside, positive outside)
// - Performance issues: Reduce grid dimensions or optimize shaders
// - Holes in mesh: May need to adjust isosurface value or grid resolution
