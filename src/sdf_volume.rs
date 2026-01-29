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
/// This type implements [`Sculptable<f32>`](crate::sculptable::Sculptable) with identity
/// conversion (values are already signed distances).
///
/// # Usage
///
/// `SdfVolume` should be spawned as a **child** of a chunk entity:
///
/// ```ignore
/// let chunk = commands.spawn((Chunk, ChunkPos(ivec3(0, 0, 0)))).id();
///
/// let mut volume = SdfVolume::new();
/// fill_centered_sphere(&mut volume, 12.0);
///
/// commands.spawn((
///     volume,
///     GenerateMesh,
///     Transform::default(),
/// )).set_parent(chunk);
/// ```
///
/// # Example
///
/// ```
/// use bevy_sculpter::prelude::*;
///
/// let mut volume = SdfVolume::new();
///
/// // Set a single voxel to be inside
/// volume.set(16, 16, 16, -1.0);
///
/// // Query a voxel
/// let distance = volume.get(16, 16, 16);
/// assert!(distance < 0.0); // Inside
/// ```
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

    // =========================================================================
    // Convenience methods
    // =========================================================================

    /// Checks if a voxel is inside the surface (negative distance).
    #[inline]
    pub fn is_inside(&self, x: u32, y: u32, z: u32) -> bool {
        self.get(x, y, z) < 0.0
    }

    /// Checks if a voxel is outside the surface (positive distance).
    #[inline]
    pub fn is_outside(&self, x: u32, y: u32, z: u32) -> bool {
        self.get(x, y, z) > 0.0
    }

    /// Checks if a voxel is on the surface (near zero distance).
    #[inline]
    pub fn is_surface(&self, x: u32, y: u32, z: u32, threshold: f32) -> bool {
        self.get(x, y, z).abs() <= threshold
    }
}

// ============================================================================
// Mesh generation marker
// ============================================================================

/// Marker component that triggers mesh generation.
///
/// # Behavior
///
/// - **On child entity**: Only that child's mesh is regenerated
/// - **On parent chunk**: Propagates to ALL children, regenerating all meshes
///
/// The component is automatically removed after mesh generation completes.
///
/// # Example
///
/// ```ignore
/// // Trigger remeshing for a single field
/// commands.entity(terrain_child).insert(GenerateMesh);
///
/// // Trigger remeshing for ALL children of a chunk
/// commands.entity(chunk_entity).insert(GenerateMesh);
/// ```
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct GenerateMesh;

// ============================================================================
// Utility types
// ============================================================================

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

// ============================================================================
// Backward compatibility
// ============================================================================

/// Alias for backward compatibility.
#[deprecated(since = "0.2.0", note = "Renamed to SdfVolume")]
pub type DensityField = SdfVolume;
