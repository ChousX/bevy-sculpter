//! Density field storage and operations for SDF-based volumetric data.
//!
//! The [`DensityField`] component stores signed distance field (SDF) values on a 3D grid.
//! Negative values represent the interior of a shape, positive values the exterior,
//! and the zero-crossing defines the surface.

use crate::{DENSITY_FIELD_SIZE, FIELD_VOLUME, field::Field};
use bevy::prelude::*;

/// A 3D grid of signed distance field (SDF) values.
///
/// Each voxel stores a floating-point density value where:
/// - **Negative** = inside the surface
/// - **Positive** = outside the surface  
/// - **Zero** = on the surface
///
/// The field is stored as a flat `Vec<f32>` in X-Y-Z order (X varies fastest).
///
/// # Example
///
/// ```
/// use bevy_sculpter::prelude::*;
///
/// let mut field = DensityField::new();
///
/// // Set a single voxel to be inside
/// field.set(16, 16, 16, -1.0);
///
/// // Query a voxel
/// let density = field.get(16, 16, 16);
/// assert!(density < 0.0); // Inside
/// ```
#[derive(Component, Clone, Deref, DerefMut, Debug)]
pub struct DensityField(pub Vec<f32>);

impl Default for DensityField {
    fn default() -> Self {
        Self(vec![1.0; FIELD_VOLUME]) // All outside
    }
}

impl Field<f32> for DensityField {
    const SIZE: UVec3 = DENSITY_FIELD_SIZE;
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

impl DensityField {
    /// Creates a new density field with all voxels set to exterior (1.0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a density field with all voxels set to the given value.
    ///
    /// # Arguments
    /// * `value` - The density value for all voxels (negative = inside, positive = outside)
    pub fn filled(value: f32) -> Self {
        Self(vec![value; FIELD_VOLUME])
    }

    // =========================================================================
    // SDF-specific operations (not part of generic Field trait)
    // =========================================================================

    /// Checks if a voxel is inside the surface (negative density).
    #[inline]
    pub fn is_inside(&self, x: u32, y: u32, z: u32) -> bool {
        self.get(x, y, z) < 0.0
    }

    /// Checks if a voxel is outside the surface (positive density).
    #[inline]
    pub fn is_outside(&self, x: u32, y: u32, z: u32) -> bool {
        self.get(x, y, z) > 0.0
    }

    /// Checks if a voxel is on the surface (near zero density).
    #[inline]
    pub fn is_surface(&self, x: u32, y: u32, z: u32, threshold: f32) -> bool {
        self.get(x, y, z).abs() <= threshold
    }

    // ... rest of the SDF-specific methods (nearest_interior, raycasting, redistancing, etc.)
    // remain unchanged ...
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

// ... rest of the file (raycasting, redistancing, etc.) remains the same ...

/// Marker component indicating this chunk needs remeshing.
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct GenerateMesh;
