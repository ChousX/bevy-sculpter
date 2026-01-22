use crate::field::Field;
use bevy::prelude::*;

/// Anything that can be meshed with Surface Nets.
///
/// Implementations convert their storage type to density values where:
/// - Negative = inside the surface
/// - Positive = outside the surface
/// - Zero = exactly on the surface
pub trait Sculptable<T: Copy + Default>: Field<T> + Component + Clone {
    /// The default iso value for out-of-bounds sampling.
    const DEFAULT_ISO: f32 = 1.0; // Outside by default

    /// Convert the storage type to a density value.
    fn to_iso(value: T) -> f32;

    /// Sample ISO-Value at grid coordinates.
    #[inline]
    fn sample(&self, x: u32, y: u32, z: u32) -> f32 {
        Self::to_iso(self.get(x, y, z))
    }

    /// Sample with signed coordinates, returns default if out of bounds.
    #[inline]
    fn sample_signed(&self, x: i32, y: i32, z: i32) -> f32 {
        self.get_signed(x, y, z)
            .map(Self::to_iso)
            .unwrap_or(Self::DEFAULT_ISO)
    }

    /// Sample at IVec3 position.
    #[inline]
    fn sample_ivec3(&self, pos: IVec3) -> f32 {
        self.sample_signed(pos.x, pos.y, pos.z)
    }

    /// Try to sample at IVec3 position, returning None if out of bounds.
    #[inline]
    fn try_sample_ivec3(&self, pos: IVec3) -> Option<f32> {
        self.get_ivec3(pos).map(Self::to_iso)
    }
}
