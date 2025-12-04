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
