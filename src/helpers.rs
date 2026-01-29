//! Sculpting brush functions for modifying SDF volumes.
//!
//! This module provides various brush types for interactive sculpting:
//!
//! - Hard CSG add/subtract operations via [`brush_sphere`]
//! - Continuous smooth sculpting via [`brush_smooth`] and [`brush_smooth_timed`]
//! - Surface smoothing via [`brush_blur`]
//! - Terrain flattening via [`brush_flatten`]
//!
//! # Coordinate System
//!
//! All brush functions operate in **grid coordinates** (0-32 for the default field size).
//! Convert from world coordinates using the mesh size:
//!
//! ```ignore
//! let scale = Vec3::splat(32.0) / mesh_size;
//! let grid_center = world_center * scale;
//! let grid_radius = world_radius * scale.x;
//! ```

use crate::{field::Field, sdf_volume::SdfVolume};
use bevy::prelude::*;

/// Fills the SDF volume with a sphere.
///
/// Sets each voxel to the signed distance from the sphere surface.
///
/// # Arguments
/// * `volume` - The SDF volume to modify
/// * `center` - Sphere center in grid coordinates
/// * `radius` - Sphere radius in grid units
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use bevy_sculpter::prelude::*;
/// use bevy_sculpter::helpers::fill_sphere;
///
/// let mut volume = SdfVolume::new();
/// fill_sphere(&mut volume, vec3(16.0, 16.0, 16.0), 10.0);
/// ```
pub fn fill_sphere(volume: &mut SdfVolume, center: Vec3, radius: f32) {
    for pos in SdfVolume::positions() {
        let pos_f = pos.as_vec3();
        volume.set(pos.x, pos.y, pos.z, pos_f.distance(center) - radius);
    }
}

/// Fills the SDF volume with a centered sphere.
///
/// Convenience wrapper for [`fill_sphere`] that places the sphere at the field center.
///
/// # Arguments
/// * `volume` - The SDF volume to modify
/// * `radius` - Sphere radius in grid units
pub fn fill_centered_sphere(volume: &mut SdfVolume, radius: f32) {
    let center = SdfVolume::SIZE.as_vec3() / 2.0;
    fill_sphere(volume, center, radius);
}

/// Applies a hard CSG sphere brush (instant add or subtract).
///
/// Uses min/max CSG operations for immediate, sharp-edged modifications.
/// For smoother, continuous sculpting, use [`brush_smooth`] instead.
///
/// # Arguments
/// * `volume` - The SDF volume to modify
/// * `center` - Brush center in grid coordinates
/// * `radius` - Brush radius in grid units
/// * `add` - If `true`, adds material (CSG union); if `false`, removes (CSG subtract)
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use bevy_sculpter::prelude::*;
/// use bevy_sculpter::helpers::brush_sphere;
///
/// let mut volume = SdfVolume::new();
///
/// // Add a sphere
/// brush_sphere(&mut volume, vec3(16.0, 16.0, 16.0), 8.0, true);
///
/// // Carve out a smaller sphere
/// brush_sphere(&mut volume, vec3(20.0, 16.0, 16.0), 4.0, false);
/// ```
pub fn brush_sphere(volume: &mut SdfVolume, center: Vec3, radius: f32, add: bool) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(SdfVolume::SIZE.as_vec3() - Vec3::ONE)
        .as_ivec3();

    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let pos = vec3(x as f32, y as f32, z as f32);
                let sphere_sdf = pos.distance(center) - radius;
                let current = volume.get(x as u32, y as u32, z as u32);

                let new_val = if add {
                    current.min(sphere_sdf)
                } else {
                    current.max(-sphere_sdf)
                };
                volume.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Applies a smooth brush that gradually adjusts SDF values.
///
/// Unlike [`brush_sphere`] which uses CSG min/max operations, this brush
/// additively changes values, allowing for continuous strokes that accumulate.
///
/// # Arguments
/// * `volume` - The SDF volume to modify
/// * `center` - Brush center in grid coordinates
/// * `radius` - Brush radius in grid units
/// * `strength` - How much to change per application (positive = remove, negative = add)
/// * `falloff` - Falloff curve power (1.0 = linear, 2.0 = quadratic, etc.)
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use bevy_sculpter::prelude::*;
/// use bevy_sculpter::helpers::brush_smooth;
///
/// let mut volume = SdfVolume::new();
///
/// // Gently add material with quadratic falloff
/// brush_smooth(&mut volume, vec3(16.0, 16.0, 16.0), 5.0, -0.5, 2.0);
/// ```
pub fn brush_smooth(
    volume: &mut SdfVolume,
    center: Vec3,
    radius: f32,
    strength: f32,
    falloff: f32,
) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(SdfVolume::SIZE.as_vec3() - Vec3::ONE)
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

                let current = volume.get(x as u32, y as u32, z as u32);
                let delta = strength * influence;
                let new_val = current + delta;

                volume.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Applies a smooth brush with time-based strength for continuous strokes.
///
/// Wrapper around [`brush_smooth`] that scales strength by delta time,
/// making the effect frame-rate independent.
///
/// # Arguments
/// * `volume` - The SDF volume to modify
/// * `center` - Brush center in grid coordinates
/// * `radius` - Brush radius in grid units
/// * `rate` - Change per second (positive = remove material, negative = add)
/// * `delta_time` - Time since last frame in seconds
/// * `falloff` - Falloff curve power (1.0 = linear, 2.0 = quadratic)
pub fn brush_smooth_timed(
    volume: &mut SdfVolume,
    center: Vec3,
    radius: f32,
    rate: f32,
    delta_time: f32,
    falloff: f32,
) {
    brush_smooth(volume, center, radius, rate * delta_time, falloff);
}

/// Applies a flatten brush that pushes values toward a target height plane.
///
/// Useful for creating flat surfaces or leveling terrain.
///
/// # Arguments
/// * `volume` - The SDF volume to modify
/// * `center` - Brush center in grid coordinates
/// * `radius` - Brush radius in grid units
/// * `target_height` - The Y coordinate to flatten toward
/// * `strength` - How strongly to push toward target (0.0-1.0 typical)
/// * `falloff` - Falloff curve power
pub fn brush_flatten(
    volume: &mut SdfVolume,
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
        .min(SdfVolume::SIZE.as_vec3() - Vec3::ONE)
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

                let current = volume.get(x as u32, y as u32, z as u32);
                let target_sdf = y as f32 - target_height;
                let new_val = current + (target_sdf - current) * influence;

                volume.set(x as u32, y as u32, z as u32, new_val);
            }
        }
    }
}

/// Applies a blur/smooth brush that averages values with neighbors.
///
/// Useful for smoothing out rough surfaces or blending brush strokes.
///
/// # Arguments
/// * `volume` - The SDF volume to modify
/// * `center` - Brush center in grid coordinates
/// * `radius` - Brush radius in grid units
/// * `strength` - Blend factor toward average (0.0-1.0)
/// * `falloff` - Falloff curve power
pub fn brush_blur(volume: &mut SdfVolume, center: Vec3, radius: f32, strength: f32, falloff: f32) {
    let min = (center - Vec3::splat(radius + 1.0))
        .max(Vec3::ZERO)
        .as_ivec3();
    let max = (center + Vec3::splat(radius + 1.0))
        .min(SdfVolume::SIZE.as_vec3() - Vec3::ONE)
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
                    if let Some(v) = volume.get_signed(x + dx, y + dy, z + dz) {
                        sum += v;
                        count += 1;
                    }
                }

                if count == 0 {
                    continue;
                }

                let avg = sum / count as f32;
                let current = volume.get(x as u32, y as u32, z as u32);

                let t = 1.0 - (dist / radius);
                let influence = t.powf(falloff) * strength;

                let new_val = current + (avg - current) * influence;
                updates.push((x as u32, y as u32, z as u32, new_val));
            }
        }
    }

    // Second pass: apply updates
    for (x, y, z, val) in updates {
        volume.set(x, y, z, val);
    }
}
