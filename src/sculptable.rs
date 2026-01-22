use crate::field::Field;

/// Anything that can be meshed with Surface Nets.
///
/// Implementations convert their storage type to density values where:
/// - Negative = inside the surface
/// - Positive = outside the surface
/// - Zero = exactly on the surface
pub trait Sculptable<T: Copy + Default>: Field<T> {
    /// Convert the storage type to a density value.
    fn to_iso(value: T) -> f32;

    /// Sample density at grid coordinates.
    #[inline]
    fn sample(&self, x: u32, y: u32, z: u32) -> f32 {
        Self::to_iso(self.get(x, y, z))
    }

    /// Sample with signed coordinates, returns None if out of bounds.
    #[inline]
    fn sample_signed(&self, x: i32, y: i32, z: i32) -> Option<f32> {
        self.get_signed(x, y, z).map(Self::to_iso)
    }

    /// Sample at IVec3 position.
    #[inline]
    fn sample_ivec3(&self, pos: IVec3) -> Option<f32> {
        self.sample_signed(pos.x, pos.y, pos.z)
    }
}
