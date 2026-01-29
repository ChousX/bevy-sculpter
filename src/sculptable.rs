//! The core trait for meshable volumetric fields.
//!
//! [`Sculptable`] is the primary trait for any field that can be meshed with Surface Nets.
//! It extends [`Field`] with the ability to convert storage values to signed distance values.

use bevy::prelude::*;

use crate::field::Field;

/// A volumetric field that can be meshed with Surface Nets.
///
/// This trait converts the underlying storage type `T` to signed distance values where:
/// - Negative = inside the surface
/// - Positive = outside the surface  
/// - Zero = exactly on the surface
///
/// # Type Parameters
/// * `T` - The storage type (e.g., `f32` for raw SDF, `u8` for binary/material IDs)
///
/// # Example: Raw SDF
///
/// ```ignore
/// impl Sculptable<f32> for SdfVolume {
///     fn to_iso(value: f32) -> f32 {
///         value // Identity - already an SDF
///     }
/// }
/// ```
///
/// # Example: Binary Voxels
///
/// ```ignore
/// impl Sculptable<u8> for BinaryVoxelField {
///     fn to_iso(value: u8) -> f32 {
///         if value > 0 { -1.0 } else { 1.0 }
///     }
/// }
/// ```
pub trait Sculptable<T>: Field<T> + Component
where
    T: Copy + Default + Send + Sync + 'static,
{
    /// The default iso value for out-of-bounds sampling.
    ///
    /// Defaults to `1.0` (outside/air), meaning out-of-bounds areas
    /// won't generate surfaces.
    const DEFAULT_ISO: f32 = 1.0;

    /// Convert the storage type to a signed distance value.
    ///
    /// This is the core conversion that determines surface boundaries:
    /// - Return negative values for "inside" the surface
    /// - Return positive values for "outside" the surface
    /// - The zero-crossing defines the surface
    fn to_iso(value: T) -> f32;

    /// Sample and convert to iso at unsigned grid coordinates.
    #[inline]
    fn sample_iso(&self, x: u32, y: u32, z: u32) -> f32 {
        Self::to_iso(self.get(x, y, z))
    }

    /// Sample and convert to iso at UVec3 coordinates.
    #[inline]
    fn sample_iso_uvec3(&self, pos: UVec3) -> f32 {
        Self::to_iso(self.get_uvec3(pos))
    }

    /// Sample and convert to iso at signed coordinates.
    ///
    /// Returns `None` if out of bounds.
    #[inline]
    fn sample_iso_signed(&self, x: i32, y: i32, z: i32) -> Option<f32> {
        self.get_signed(x, y, z).map(Self::to_iso)
    }

    /// Sample and convert to iso at IVec3 coordinates.
    #[inline]
    fn sample_iso_ivec3(&self, pos: IVec3) -> Option<f32> {
        self.sample_iso_signed(pos.x, pos.y, pos.z)
    }

    /// Check if a voxel is inside the surface (iso < 0).
    #[inline]
    fn is_inside(&self, x: u32, y: u32, z: u32) -> bool {
        self.sample_iso(x, y, z) < 0.0
    }

    /// Check if a voxel is outside the surface (iso > 0).
    #[inline]
    fn is_outside(&self, x: u32, y: u32, z: u32) -> bool {
        self.sample_iso(x, y, z) > 0.0
    }

    /// Check if a voxel is near the surface (|iso| < threshold).
    #[inline]
    fn is_surface(&self, x: u32, y: u32, z: u32, threshold: f32) -> bool {
        self.sample_iso(x, y, z).abs() < threshold
    }
}

/// Extension trait for SDF-specific operations like gradient computation.
///
/// Automatically implemented for any `Sculptable<f32>`.
pub trait SdfOps: Sculptable<f32> {
    /// Compute the gradient (normal direction) at a point using central differences.
    ///
    /// Returns a normalized vector pointing away from the surface (toward positive values).
    fn gradient(&self, pos: IVec3) -> Vec3 {
        let dx = self
            .sample_iso_signed(pos.x + 1, pos.y, pos.z)
            .unwrap_or(Self::DEFAULT_ISO)
            - self
                .sample_iso_signed(pos.x - 1, pos.y, pos.z)
                .unwrap_or(Self::DEFAULT_ISO);
        let dy = self
            .sample_iso_signed(pos.x, pos.y + 1, pos.z)
            .unwrap_or(Self::DEFAULT_ISO)
            - self
                .sample_iso_signed(pos.x, pos.y - 1, pos.z)
                .unwrap_or(Self::DEFAULT_ISO);
        let dz = self
            .sample_iso_signed(pos.x, pos.y, pos.z + 1)
            .unwrap_or(Self::DEFAULT_ISO)
            - self
                .sample_iso_signed(pos.x, pos.y, pos.z - 1)
                .unwrap_or(Self::DEFAULT_ISO);

        let grad = vec3(dx, dy, dz);
        if grad.length_squared() > 0.0001 {
            grad.normalize()
        } else {
            Vec3::Y
        }
    }

    /// Compute the gradient at UVec3 coordinates.
    #[inline]
    fn gradient_uvec3(&self, pos: UVec3) -> Vec3 {
        self.gradient(pos.as_ivec3())
    }
}

// Blanket implementation: any Sculptable<f32> gets SdfOps
impl<F: Sculptable<f32>> SdfOps for F {}
