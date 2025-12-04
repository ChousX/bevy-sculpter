use crate::{
    DENSITY_FIELD_SIZE, FIELD_VOLUME, density_field::DensityField, mesher::DensityFieldMeshSize,
};
use bevy::prelude::*;

pub fn fill_sphere(density_field: &mut DensityField, center: Vec3, radius: f32) {
    for z in 0..DENSITY_FIELD_SIZE.z {
        for y in 0..DENSITY_FIELD_SIZE.y {
            for x in 0..DENSITY_FIELD_SIZE.x {
                let pos = vec3(x as f32, y as f32, z as f32);
                density_field.set(x, y, z, pos.distance(center) - radius);
            }
        }
    }
}

pub fn fill_centered_sphere(density_field: &mut DensityField, radius: f32) {
    let center = DENSITY_FIELD_SIZE.as_vec3() / 2.0;
    fill_sphere(density_field, center, radius);
}

pub fn brush_sphere(density_field: &mut DensityField, center: Vec3, radius: f32, add: bool) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(DENSITY_FIELD_SIZE.as_vec3() - Vec3::ONE)
        .as_ivec3();

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let pos = vec3(x as f32, y as f32, z as f32);
                let sphere_sdf = pos.distance(center) - radius;
                let current = density_field.get(x as u32, y as u32, z as u32);

                let new_val = if add {
                    current.min(sphere_sdf)
                } else {
                    current.max(-sphere_sdf)
                };
                density_field.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Smooth brush that gradually adjusts density values over time.
///
/// Unlike `brush_sphere` which uses CSG min/max operations, this brush
/// additively changes density values, allowing for continuous strokes
/// that accumulate effect over time.
///
/// # Arguments
/// * `density_field` - The field to modify
/// * `center` - Brush center in grid coordinates
/// * `radius` - Brush radius in grid units
/// * `strength` - How much to change density per application (positive = remove, negative = add)
/// * `falloff` - Falloff curve power (1.0 = linear, 2.0 = quadratic, etc.)
pub fn brush_smooth(
    density_field: &mut DensityField,
    center: Vec3,
    radius: f32,
    strength: f32,
    falloff: f32,
) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(DENSITY_FIELD_SIZE.as_vec3() - Vec3::ONE)
        .as_ivec3();

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let pos = vec3(x as f32, y as f32, z as f32);
                let dist = pos.distance(center);

                if dist > radius {
                    continue;
                }

                // Calculate falloff: 1.0 at center, 0.0 at edge
                let t = 1.0 - (dist / radius);
                let influence = t.powf(falloff);

                let current = density_field.get(x as u32, y as u32, z as u32);
                let delta = strength * influence;
                let new_val = current + delta;

                density_field.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Smooth brush with time-based strength for continuous strokes.
///
/// Wrapper around `brush_smooth` that scales strength by delta time,
/// making the effect frame-rate independent.
///
/// # Arguments
/// * `density_field` - The field to modify
/// * `center` - Brush center in grid coordinates  
/// * `radius` - Brush radius in grid units
/// * `rate` - Density change per second (positive = remove material, negative = add)
/// * `delta_time` - Time since last frame in seconds
/// * `falloff` - Falloff curve power (1.0 = linear, 2.0 = quadratic)
pub fn brush_smooth_timed(
    density_field: &mut DensityField,
    center: Vec3,
    radius: f32,
    rate: f32,
    delta_time: f32,
    falloff: f32,
) {
    brush_smooth(density_field, center, radius, rate * delta_time, falloff);
}

/// Flatten brush - pushes density values toward a target height plane.
///
/// Useful for creating flat surfaces or leveling terrain.
///
/// # Arguments
/// * `density_field` - The field to modify
/// * `center` - Brush center in grid coordinates
/// * `radius` - Brush radius in grid units
/// * `target_height` - The Y coordinate to flatten toward
/// * `strength` - How strongly to push toward target (0.0-1.0 typical)
/// * `falloff` - Falloff curve power
pub fn brush_flatten(
    density_field: &mut DensityField,
    center: Vec3,
    radius: f32,
    target_height: f32,
    strength: f32,
    falloff: f32,
) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(DENSITY_FIELD_SIZE.as_vec3() - Vec3::ONE)
        .as_ivec3();

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let pos = vec3(x as f32, y as f32, z as f32);
                let dist = pos.distance(center);

                if dist > radius {
                    continue;
                }

                let t = 1.0 - (dist / radius);
                let influence = t.powf(falloff) * strength;

                let current = density_field.get(x as u32, y as u32, z as u32);
                // Target SDF: negative below target_height, positive above
                let target_sdf = y as f32 - target_height;
                let new_val = current + (target_sdf - current) * influence;

                density_field.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Smooth/blur brush - averages density values with neighbors.
///
/// Useful for smoothing out rough surfaces or blending brush strokes.
///
/// # Arguments
/// * `density_field` - The field to modify
/// * `center` - Brush center in grid coordinates
/// * `radius` - Brush radius in grid units
/// * `strength` - Blend factor toward average (0.0-1.0)
/// * `falloff` - Falloff curve power
pub fn brush_blur(
    density_field: &mut DensityField,
    center: Vec3,
    radius: f32,
    strength: f32,
    falloff: f32,
) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(DENSITY_FIELD_SIZE.as_vec3() - Vec3::ONE)
        .as_ivec3();

    // First pass: compute averages (read-only)
    let mut updates: Vec<(u32, u32, u32, f32)> = Vec::new();

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let pos = vec3(x as f32, y as f32, z as f32);
                let dist = pos.distance(center);

                if dist > radius {
                    continue;
                }

                // Sample 6 neighbors
                let mut sum = 0.0;
                let mut count = 0;

                for (dx, dy, dz) in [
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    if let Some(v) = density_field.get_signed(x + dx, y + dy, z + dz) {
                        sum += v;
                        count += 1;
                    }
                }

                if count == 0 {
                    continue;
                }

                let avg = sum / count as f32;
                let current = density_field.get(x as u32, y as u32, z as u32);

                let t = 1.0 - (dist / radius);
                let influence = t.powf(falloff) * strength;

                let new_val = current + (avg - current) * influence;
                updates.push((x as u32, y as u32, z as u32, new_val));
            }
        }
    }

    // Second pass: apply updates
    for (x, y, z, val) in updates {
        density_field.set(x, y, z, val);
    }
}
