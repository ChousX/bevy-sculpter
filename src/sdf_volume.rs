//! Signed distance field volume storage.
//!
//! The [`SdfVolume`] component stores signed distance field (SDF) values on a 3D grid.
//! Negative values represent the interior of a shape, positive values the exterior,
//! and the zero-crossing defines the surface.

use crate::{FIELD_SIZE, FIELD_VOLUME, field::Field};
use bevy::prelude::*;

/// A 3D grid of signed distance field (SDF) values.
///
/// Each voxel stores a floating-point distance value where:
/// - **Negative** = inside the surface
/// - **Positive** = outside the surface  
/// - **Zero** = on the surface
///
/// `SdfVolume` should be spawned as a **child** of a chunk entity:
#[derive(Component, Clone, Deref, DerefMut, Debug)]
pub struct SdfVolume(pub Vec<f32>);

impl Default for SdfVolume {
    fn default() -> Self {
        Self(vec![1.0; FIELD_VOLUME]) // All outside
    }
}

impl Field<f32> for SdfVolume {
    const SIZE: UVec3 = FIELD_SIZE;
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

impl SdfVolume {
    /// Creates a new SDF volume with all voxels set to exterior (1.0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an SDF volume with all voxels set to the given value.
    ///
    /// # Arguments
    /// * `value` - The distance value for all voxels (negative = inside, positive = outside)
    pub fn filled(value: f32) -> Self {
        Self(vec![value; FIELD_VOLUME])
    }
}

/// Marker component that triggers mesh generation.
///
/// # Behavior
///
/// - **On child entity**: Only that child's mesh is regenerated
/// - **On parent chunk**: Propagates to ALL children, regenerating all meshes
///
/// The component is automatically removed after mesh generation completes.
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct GenerateMesh;

/// Result of finding the nearest interior point in an SDF volume.
#[derive(Clone, Copy, Debug)]
pub struct NearestInteriorResult {
    /// Grid coordinates of the nearest interior voxel.
    pub grid_pos: UVec3,
    /// World-space position of the nearest interior voxel center.
    pub world_pos: Vec3,
    /// The distance value at this point (negative = inside).
    pub distance: f32,
    /// Squared distance from query point to this voxel (in grid space).
    pub distance_sq: f32,
}

/// Alias for backward compatibility.
#[deprecated(since = "0.2.0", note = "Renamed to SdfVolume")]
pub type DensityField = SdfVolume;
