use crate::{DENSITY_FIELD_SIZE, FIELD_VOLUME};
use bevy::prelude::*;

/// The density field (SDF). Negative = inside, Positive = outside.
#[derive(Component, Clone, Deref, DerefMut, Debug)]
pub struct DensityField(pub Vec<f32>);

impl Default for DensityField {
    fn default() -> Self {
        Self(vec![1.0; FIELD_VOLUME]) // All outside
    }
}

impl DensityField {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn filled(value: f32) -> Self {
        Self(vec![value; FIELD_VOLUME])
    }

    #[inline]
    pub fn index(x: u32, y: u32, z: u32) -> usize {
        (x + y * DENSITY_FIELD_SIZE.x + z * DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.y) as usize
    }

    #[inline]
    pub fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < DENSITY_FIELD_SIZE.x
            && (y as u32) < DENSITY_FIELD_SIZE.y
            && (z as u32) < DENSITY_FIELD_SIZE.z
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        if x < DENSITY_FIELD_SIZE.x && y < DENSITY_FIELD_SIZE.y && z < DENSITY_FIELD_SIZE.z {
            self.0[Self::index(x, y, z)] = value;
        }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        if x < DENSITY_FIELD_SIZE.x && y < DENSITY_FIELD_SIZE.y && z < DENSITY_FIELD_SIZE.z {
            self.0[Self::index(x, y, z)]
        } else {
            1.0 // Outside = exterior
        }
    }

    /// Get with signed coords (for neighbor sampling)
    #[inline]
    pub fn get_signed(&self, x: i32, y: i32, z: i32) -> Option<f32> {
        if Self::in_bounds(x, y, z) {
            Some(self.0[Self::index(x as u32, y as u32, z as u32)])
        } else {
            None
        }
    }

    
}

/// Marker: this chunk needs remeshing
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct DensityFieldDirty;
