//! Sculpting brush functions for modifying density fields.
//!
//! This module provides various brush types for interactive sculpting:
//!
//! - Hard CSG add/subtract operations via [`brush_sphere`]
//! - Continuous smooth sculpting via [`brush_smooth`] and [`brush_smooth_timed`]
//! - Surface smoothing via [`brush_blur`]
//! - Terrain flattening via [`brush_flatten`]
//!
//! All functions are generic over any type implementing [`Field<f32>`].

use crate::field::Field;
use bevy::prelude::*;

/// Fills the density field with a sphere SDF.
///
/// Sets each voxel to the signed distance from the sphere surface.
pub fn fill_sphere<F: Field<f32>>(field: &mut F, center: Vec3, radius: f32) {
    for pos in F::positions() {
        let dist = pos.as_vec3().distance(center) - radius;
        field.set(pos.x, pos.y, pos.z, dist);
    }
}

/// Fills the density field with a centered sphere SDF.
pub fn fill_centered_sphere<F: Field<f32>>(field: &mut F, radius: f32) {
    let center = F::SIZE.as_vec3() / 2.0;
    fill_sphere(field, center, radius);
}

/// Applies a hard CSG sphere brush (instant add or subtract).
///
/// Uses min/max CSG operations for immediate, sharp-edged modifications.
pub fn brush_sphere<F: Field<f32>>(field: &mut F, center: Vec3, radius: f32, add: bool) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(F::SIZE.as_vec3() - Vec3::ONE)
        .as_ivec3();

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let pos = vec3(x as f32, y as f32, z as f32);
                let sphere_sdf = pos.distance(center) - radius;
                let current = field.get(x as u32, y as u32, z as u32);

                let new_val = if add {
                    current.min(sphere_sdf)
                } else {
                    current.max(-sphere_sdf)
                };
                field.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Applies a smooth brush that gradually adjusts density values.
///
/// Unlike [`brush_sphere`] which uses CSG min/max operations, this brush
/// additively changes density values, allowing for continuous strokes.
pub fn brush_smooth<F: Field<f32>>(
    field: &mut F,
    center: Vec3,
    radius: f32,
    strength: f32,
    falloff: f32,
) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(F::SIZE.as_vec3() - Vec3::ONE)
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
                let influence = t.powf(falloff);

                let current = field.get(x as u32, y as u32, z as u32);
                let new_val = current + strength * influence;
                field.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Applies a smooth brush with time-based strength for continuous strokes.
pub fn brush_smooth_timed<F: Field<f32>>(
    field: &mut F,
    center: Vec3,
    radius: f32,
    rate: f32,
    delta_time: f32,
    falloff: f32,
) {
    brush_smooth(field, center, radius, rate * delta_time, falloff);
}

/// Applies a flatten brush that pushes density values toward a target height plane.
pub fn brush_flatten<F: Field<f32>>(
    field: &mut F,
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
        .min(F::SIZE.as_vec3() - Vec3::ONE)
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

                let current = field.get(x as u32, y as u32, z as u32);
                let target_sdf = y as f32 - target_height;
                let new_val = current + (target_sdf - current) * influence;
                field.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Applies a blur/smooth brush that averages density values with neighbors.
pub fn brush_blur<F: Field<f32>>(
    field: &mut F,
    center: Vec3,
    radius: f32,
    strength: f32,
    falloff: f32,
) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(F::SIZE.as_vec3() - Vec3::ONE)
        .as_ivec3();

    // First pass: compute averages
    let mut updates: Vec<(u32, u32, u32, f32)> = Vec::new();

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let pos = vec3(x as f32, y as f32, z as f32);
                let dist = pos.distance(center);

                if dist > radius {
                    continue;
                }

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
                    if let Some(v) = field.get_signed(x + dx, y + dy, z + dz) {
                        sum += v;
                        count += 1;
                    }
                }

                if count == 0 {
                    continue;
                }

                let avg = sum / count as f32;
                let current = field.get(x as u32, y as u32, z as u32);

                let t = 1.0 - (dist / radius);
                let influence = t.powf(falloff) * strength;

                let new_val = current + (avg - current) * influence;
                updates.push((x as u32, y as u32, z as u32, new_val));
            }
        }
    }

    // Second pass: apply updates
    for (x, y, z, val) in updates {
        field.set(x, y, z, val);
    }
}
