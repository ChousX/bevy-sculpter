//! Generic field trait for 3D voxel data storage.
//!
//! This module provides the [`Field`] trait which abstracts over different
//! types of volumetric data (density, materials, etc.) with a common interface.

use bevy::prelude::*;

/// A 3D grid of voxel data with a fixed size.
///
/// This trait provides a common interface for volumetric data storage,
/// including density fields (f32), material fields (u8), and any other
/// per-voxel data types.
///
/// # Type Parameters
/// * `T` - The element type stored at each voxel
///
/// # Coordinate System
/// Uses X-Y-Z ordering where X varies fastest in the underlying storage.
pub trait Field<T: Copy + Default>: Default {
    /// The size of the field grid as a `UVec3`.
    const SIZE: UVec3;

    /// The total number of voxels in the field.
    const VOLUME: usize = (Self::SIZE.x * Self::SIZE.y * Self::SIZE.z) as usize;

    /// The default value for out-of-bounds access.
    const DEFAULT: T;

    /// Returns a reference to the underlying data slice.
    fn data(&self) -> &[T];

    /// Returns a mutable reference to the underlying data slice.
    fn data_mut(&mut self) -> &mut [T];

    /// Computes the flat array index for a given (x, y, z) coordinate.
    ///
    /// Uses X-Y-Z ordering where X varies fastest.
    #[inline]
    fn index(x: u32, y: u32, z: u32) -> usize {
        (x + y * Self::SIZE.x + z * Self::SIZE.x * Self::SIZE.y) as usize
    }

    /// Converts a flat index back to (x, y, z) coordinates.
    #[inline]
    fn coords(index: usize) -> UVec3 {
        let index = index as u32;
        let z = index / (Self::SIZE.x * Self::SIZE.y);
        let rem = index % (Self::SIZE.x * Self::SIZE.y);
        let y = rem / Self::SIZE.x;
        let x = rem % Self::SIZE.x;
        uvec3(x, y, z)
    }

    /// Checks if unsigned coordinates are within the field bounds.
    #[inline]
    fn in_bounds_unsigned(x: u32, y: u32, z: u32) -> bool {
        x < Self::SIZE.x && y < Self::SIZE.y && z < Self::SIZE.z
    }

    /// Checks if signed coordinates are within the field bounds.
    ///
    /// Useful when sampling neighbors that might be outside the grid.
    #[inline]
    fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < Self::SIZE.x
            && (y as u32) < Self::SIZE.y
            && (z as u32) < Self::SIZE.z
    }

    /// Gets the value at the given unsigned coordinates.
    ///
    /// Returns [`Self::DEFAULT`] for out-of-bounds coordinates.
    #[inline]
    fn get(&self, x: u32, y: u32, z: u32) -> T {
        if Self::in_bounds_unsigned(x, y, z) {
            self.data()[Self::index(x, y, z)]
        } else {
            Self::DEFAULT
        }
    }

    /// Gets the value at the given `UVec3` coordinates.
    #[inline]
    fn get_uvec3(&self, pos: UVec3) -> T {
        self.get(pos.x, pos.y, pos.z)
    }

    /// Gets the value using signed coordinates.
    ///
    /// Returns `None` for out-of-bounds coordinates, useful for neighbor sampling.
    #[inline]
    fn get_signed(&self, x: i32, y: i32, z: i32) -> Option<T> {
        if Self::in_bounds(x, y, z) {
            Some(self.data()[Self::index(x as u32, y as u32, z as u32)])
        } else {
            None
        }
    }

    /// Gets the value at the given `IVec3` coordinates.
    ///
    /// Returns `None` for out-of-bounds coordinates.
    #[inline]
    fn get_ivec3(&self, pos: IVec3) -> Option<T> {
        self.get_signed(pos.x, pos.y, pos.z)
    }

    /// Sets the value at the given unsigned coordinates.
    ///
    /// Silently ignores out-of-bounds coordinates.
    #[inline]
    fn set(&mut self, x: u32, y: u32, z: u32, value: T) {
        if Self::in_bounds_unsigned(x, y, z) {
            self.data_mut()[Self::index(x, y, z)] = value;
        }
    }

    /// Sets the value at the given `UVec3` coordinates.
    #[inline]
    fn set_uvec3(&mut self, pos: UVec3, value: T) {
        self.set(pos.x, pos.y, pos.z, value);
    }

    /// Sets the value using signed coordinates.
    ///
    /// Returns `true` if the value was set, `false` if out of bounds.
    #[inline]
    fn set_signed(&mut self, x: i32, y: i32, z: i32, value: T) -> bool {
        if Self::in_bounds(x, y, z) {
            self.data_mut()[Self::index(x as u32, y as u32, z as u32)] = value;
            true
        } else {
            false
        }
    }

    /// Sets the value at the given `IVec3` coordinates.
    ///
    /// Returns `true` if the value was set, `false` if out of bounds.
    #[inline]
    fn set_ivec3(&mut self, pos: IVec3, value: T) -> bool {
        self.set_signed(pos.x, pos.y, pos.z, value)
    }

    /// Fills the entire field with a single value.
    #[inline]
    fn fill(&mut self, value: T) {
        self.data_mut().fill(value);
    }

    /// Returns an iterator over all (position, value) pairs.
    fn iter(&self) -> FieldIter<'_, T, Self>
    where
        Self: Sized,
    {
        FieldIter {
            field: self,
            index: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns an iterator over all positions in the field.
    fn positions() -> FieldPositionIter {
        FieldPositionIter {
            size: Self::SIZE,
            index: 0,
        }
    }
}

/// Iterator over (position, value) pairs in a field.
pub struct FieldIter<'a, T: Copy + Default, F: Field<T>> {
    field: &'a F,
    index: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: Copy + Default, F: Field<T>> Iterator for FieldIter<'a, T, F> {
    type Item = (UVec3, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < F::VOLUME {
            let pos = F::coords(self.index);
            let value = self.field.data()[self.index];
            self.index += 1;
            Some((pos, value))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = F::VOLUME - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy + Default, F: Field<T>> ExactSizeIterator for FieldIter<'a, T, F> {}

/// Iterator over all positions in a field.
pub struct FieldPositionIter {
    size: UVec3,
    index: u32,
}

impl Iterator for FieldPositionIter {
    type Item = UVec3;

    fn next(&mut self) -> Option<Self::Item> {
        let volume = self.size.x * self.size.y * self.size.z;
        if self.index < volume {
            let z = self.index / (self.size.x * self.size.y);
            let rem = self.index % (self.size.x * self.size.y);
            let y = rem / self.size.x;
            let x = rem % self.size.x;
            self.index += 1;
            Some(uvec3(x, y, z))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let volume = (self.size.x * self.size.y * self.size.z) as usize;
        let remaining = volume - self.index as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FieldPositionIter {}

/// Extension trait for fields that support spherical operations.
pub trait FieldSphereOps<T: Copy + Default>: Field<T> {
    /// Applies an operation to all voxels within a sphere.
    ///
    /// # Arguments
    /// * `center` - Center of the sphere in grid coordinates
    /// * `radius` - Radius in grid units
    /// * `op` - Operation to apply: receives (current_value, distance_from_center) and returns new value
    fn apply_sphere<F>(&mut self, center: Vec3, radius: f32, op: F)
    where
        F: Fn(T, f32) -> T,
    {
        let min = (center - Vec3::splat(radius + 1.0))
            .max(Vec3::ZERO)
            .as_ivec3();
        let max = (center + Vec3::splat(radius + 1.0))
            .min(Self::SIZE.as_vec3() - Vec3::ONE)
            .as_ivec3();

        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let pos = vec3(x as f32, y as f32, z as f32);
                    let dist = pos.distance(center);
                    if dist <= radius {
                        let current = self.get(x as u32, y as u32, z as u32);
                        let new_value = op(current, dist);
                        self.set(x as u32, y as u32, z as u32, new_value);
                    }
                }
            }
        }
    }

    /// Fills a sphere with a constant value.
    fn fill_sphere(&mut self, center: Vec3, radius: f32, value: T) {
        self.apply_sphere(center, radius, |_, _| value);
    }
}

/// Extension trait for fields that support box operations.
pub trait FieldBoxOps<T: Copy + Default>: Field<T> {
    /// Applies an operation to all voxels within an axis-aligned box.
    ///
    /// # Arguments
    /// * `min` - Minimum corner (inclusive)
    /// * `max` - Maximum corner (inclusive)
    /// * `op` - Operation to apply: receives current value and returns new value
    fn apply_box<F>(&mut self, min: IVec3, max: IVec3, op: F)
    where
        F: Fn(T) -> T,
    {
        let min = min.max(IVec3::ZERO);
        let max = max.min(Self::SIZE.as_ivec3() - IVec3::ONE);

        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let current = self.get(x as u32, y as u32, z as u32);
                    let new_value = op(current);
                    self.set(x as u32, y as u32, z as u32, new_value);
                }
            }
        }
    }

    /// Fills a box with a constant value.
    fn fill_box(&mut self, min: IVec3, max: IVec3, value: T) {
        self.apply_box(min, max, |_| value);
    }
}

// Blanket implementations for all Field types
impl<T: Copy + Default, F: Field<T>> FieldSphereOps<T> for F {}
impl<T: Copy + Default, F: Field<T>> FieldBoxOps<T> for F {}

#[cfg(test)]
mod tests {
    use super::*;

    // Test implementation
    #[derive(Default)]
    struct TestField(Vec<f32>);

    impl Field<f32> for TestField {
        const SIZE: UVec3 = uvec3(8, 8, 8);
        const DEFAULT: f32 = 0.0;

        fn data(&self) -> &[f32] {
            &self.0
        }

        fn data_mut(&mut self) -> &mut [f32] {
            &mut self.0
        }
    }

    #[test]
    fn test_index_coords_roundtrip() {
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let idx = TestField::index(x, y, z);
                    let coords = TestField::coords(idx);
                    assert_eq!(coords, uvec3(x, y, z));
                }
            }
        }
    }

    #[test]
    fn test_get_set() {
        let mut field = TestField(vec![0.0; 512]);
        field.set(3, 4, 5, 42.0);
        assert_eq!(field.get(3, 4, 5), 42.0);
        assert_eq!(field.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_out_of_bounds() {
        let field = TestField(vec![1.0; 512]);
        assert_eq!(field.get(100, 0, 0), 0.0); // DEFAULT
        assert_eq!(field.get_signed(-1, 0, 0), None);
    }

    #[test]
    fn test_fill_sphere() {
        let mut field = TestField(vec![0.0; 512]);
        field.fill_sphere(vec3(4.0, 4.0, 4.0), 2.0, 1.0);
        assert_eq!(field.get(4, 4, 4), 1.0); // Center
        assert_eq!(field.get(0, 0, 0), 0.0); // Outside
    }

    #[test]
    fn test_iter() {
        let field = TestField(vec![0.0; 512]);
        assert_eq!(field.iter().count(), 512);
    }
}
