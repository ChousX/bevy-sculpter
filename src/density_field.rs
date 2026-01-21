//! Density field storage and operations for SDF-based volumetric data.
//!
//! The [`DensityField`] component stores signed distance field (SDF) values on a 3D grid.
//! Negative values represent the interior of a shape, positive values the exterior,
//! and the zero-crossing defines the surface.

use crate::field::Field;
use bevy::prelude::*;

#[derive(Component, Clone, Deref, DerefMut, Debug)]
pub struct DefaultIsoField(pub Vec<f32>);

impl Default for DefaultIsoField {
    fn default() -> Self {
        Self(vec![1.0; Self::VOLUME]) // All outside
    }
}

impl Field<f32> for DefaultIsoField {
    const SIZE: UVec3 = uvec3(32, 32, 32);
    const DEFAULT: f32 = 1.0; // Outside = exterior

    #[inline]
    fn data(&self) -> &[f32] {
        &self.0
    }

    #[inline]
    fn data_mut(&mut self) -> &mut [f32] {
        &mut self.0
    }
}

trait IsoField {
    /// Creates a new density field with all voxels set to exterior (1.0).
    // =========================================================================
    // SDF-specific operations (not part of generic Field trait)
    // =========================================================================

    /// Checks if a voxel is inside the surface (negative density).
    #[inline]
    fn is_inside(&self, x: u32, y: u32, z: u32) -> bool;

    /// Checks if a voxel is outside the surface (positive density).
    #[inline]
    fn is_outside(&self, x: u32, y: u32, z: u32) -> bool;

    /// Checks if a voxel is on the surface (near zero density).
    #[inline]
    fn is_surface(&self, x: u32, y: u32, z: u32, threshold: f32) -> bool;
}

impl IsoField for DefaultIsoField {
    #[inline]
    fn is_inside(&self, x: u32, y: u32, z: u32) -> bool {
        self.get(x, y, z) < 0.0
    }

    #[inline]
    fn is_outside(&self, x: u32, y: u32, z: u32) -> bool {
        self.get(x, y, z) > 0.0
    }

    #[inline]
    fn is_surface(&self, x: u32, y: u32, z: u32, threshold: f32) -> bool {
        self.get(x, y, z).abs() <= threshold
    }
}

/// Result of finding the nearest interior point in a density field.
#[derive(Clone, Copy, Debug)]
pub struct NearestInteriorResult {
    /// Grid coordinates of the nearest interior voxel.
    pub grid_pos: UVec3,
    /// World-space position of the nearest interior voxel center.
    pub world_pos: Vec3,
    /// The density value at this point (negative = inside).
    pub density: f32,
    /// Squared distance from query point to this voxel (in grid space).
    pub distance_sq: f32,
}

/// Marker component indicating this chunk needs remeshing.
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct GenerateMesh;
