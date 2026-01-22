//! Density field storage and operations for SDF-based volumetric data.
//!
//! This module provides [`DefaultIsoField`], a standard 32³ density field
//! for SDF-based voxel sculpting. For custom sizes or storage types,
//! implement [`Field`] and [`Sculptable`] on your own type.

use crate::{field::Field, sculptable::Sculptable};
use bevy::prelude::*;

/// Default 32³ density field storing raw f32 SDF values.
///
/// This is the standard field type used by [`SurfaceNetsPlugin`].
/// Values follow SDF conventions:
/// - Negative = inside the surface
/// - Positive = outside the surface
/// - Zero = on the surface
///
/// # Example
/// ```ignore
/// let mut field = DefaultIsoField::new();
///
/// // Fill with a sphere SDF
/// helpers::fill_centered_sphere(&mut field, 12.0);
///
/// // Carve out a hole
/// helpers::brush_sphere(&mut field, vec3(16.0, 16.0, 20.0), 5.0, false);
/// ```
#[derive(Component, Clone, Deref, DerefMut, Debug)]
pub struct DefaultIsoField(pub Vec<f32>);

impl DefaultIsoField {
    /// Creates a new field filled with exterior values (all outside).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a field filled with a specific SDF value.
    ///
    /// Use negative values to create a solid block, positive for empty space.
    pub fn filled(value: f32) -> Self {
        Self(vec![value; Self::VOLUME])
    }

    /// Creates a field filled with interior values (all inside/solid).
    pub fn solid() -> Self {
        Self::filled(-1.0)
    }

    /// Creates a field filled with interior values (all outside/empty).
    pub fn empty() -> Self {
        Self::filled(1.0)
    }
}

impl Default for DefaultIsoField {
    fn default() -> Self {
        Self(vec![1.0; Self::VOLUME]) // All outside/empty
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

impl Sculptable<f32> for DefaultIsoField {
    fn to_iso(value: f32) -> f32 {
        value // Identity - already stores SDF values
    }
}

/// Marker component indicating this chunk needs remeshing.
///
/// Add this to any entity with a sculptable field to trigger mesh generation.
/// The component is automatically removed after meshing completes.
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct GenerateMesh;

// =============================================================================
// Example Custom Fields
// =============================================================================

/// A memory-efficient binary field using `u8` storage.
///
/// Each voxel is either solid (1) or empty (0). Useful when you don't need
/// smooth SDF gradients and want to save memory.
///
/// # Registration
/// ```ignore
/// app.register_sculptable_field::<u8, BinaryField, MyChunkManager>();
/// ```
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
