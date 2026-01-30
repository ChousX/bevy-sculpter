//! Neighbor chunk data for seamless mesh boundaries.
//!
//! When meshing a chunk, the Surface Nets algorithm needs to sample density values
//! slightly beyond the chunk boundaries to properly connect vertices at the edges.
//! This module provides structures to cache and access that neighbor data.

use bevy::prelude::*;

use crate::field::Field;

/// How many planes of neighbor data to store.
pub const NEIGHBOR_DEPTH: u32 = 2;

/// Identifies a face of a chunk for neighbor lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeighborFace {
    NegX = 0,
    PosX = 1,
    NegY = 2,
    PosY = 3,
    NegZ = 4,
    PosZ = 5,
}

impl NeighborFace {
    pub const ALL: [Self; 6] = [
        Self::NegX,
        Self::PosX,
        Self::NegY,
        Self::PosY,
        Self::NegZ,
        Self::PosZ,
    ];

    /// Returns the chunk coordinate offset for this face direction.
    pub fn offset(&self) -> IVec3 {
        match self {
            Self::NegX => ivec3(-1, 0, 0),
            Self::PosX => ivec3(1, 0, 0),
            Self::NegY => ivec3(0, -1, 0),
            Self::PosY => ivec3(0, 1, 0),
            Self::NegZ => ivec3(0, 0, -1),
            Self::PosZ => ivec3(0, 0, 1),
        }
    }

    /// Returns the (size_a, size_b) dimensions for a slice on this face.
    #[inline]
    pub fn slice_dimensions(&self, field_size: UVec3) -> (u32, u32) {
        match self {
            Self::NegX | Self::PosX => (field_size.y, field_size.z),
            Self::NegY | Self::PosY => (field_size.x, field_size.z),
            Self::NegZ | Self::PosZ => (field_size.x, field_size.y),
        }
    }

    /// Convert slice coordinates (a, b, depth) to field coordinates (x, y, z).
    #[inline]
    pub fn to_field_coords(
        &self,
        a: u32,
        b: u32,
        depth: u32,
        field_size: UVec3,
    ) -> (u32, u32, u32) {
        match self {
            Self::NegX => (field_size.x.saturating_sub(1 + depth), a, b),
            Self::PosX => (depth.min(field_size.x - 1), a, b),
            Self::NegY => (a, field_size.y.saturating_sub(1 + depth), b),
            Self::PosY => (a, depth.min(field_size.y - 1), b),
            Self::NegZ => (a, b, field_size.z.saturating_sub(1 + depth)),
            Self::PosZ => (a, b, depth.min(field_size.z - 1)),
        }
    }

    /// Check if a voxel coordinate is in this neighbor's territory.
    ///
    /// Returns `Some((a, b, depth))` if the voxel is in this neighbor's region.
    #[inline]
    pub fn voxel_to_slice_coords(
        &self,
        voxel: IVec3,
        field_size: IVec3,
    ) -> Option<(u32, u32, u32)> {
        match self {
            Self::NegX
                if voxel.x < 0
                    && voxel.y >= 0
                    && voxel.z >= 0
                    && voxel.y < field_size.y
                    && voxel.z < field_size.z =>
            {
                Some((voxel.y as u32, voxel.z as u32, (-1 - voxel.x) as u32))
            }
            Self::PosX
                if voxel.x >= field_size.x
                    && voxel.y >= 0
                    && voxel.z >= 0
                    && voxel.y < field_size.y
                    && voxel.z < field_size.z =>
            {
                Some((
                    voxel.y as u32,
                    voxel.z as u32,
                    (voxel.x - field_size.x) as u32,
                ))
            }
            Self::NegY
                if voxel.y < 0
                    && voxel.x >= 0
                    && voxel.z >= 0
                    && voxel.x < field_size.x
                    && voxel.z < field_size.z =>
            {
                Some((voxel.x as u32, voxel.z as u32, (-1 - voxel.y) as u32))
            }
            Self::PosY
                if voxel.y >= field_size.y
                    && voxel.x >= 0
                    && voxel.z >= 0
                    && voxel.x < field_size.x
                    && voxel.z < field_size.z =>
            {
                Some((
                    voxel.x as u32,
                    voxel.z as u32,
                    (voxel.y - field_size.y) as u32,
                ))
            }
            Self::NegZ
                if voxel.z < 0
                    && voxel.x >= 0
                    && voxel.y >= 0
                    && voxel.x < field_size.x
                    && voxel.y < field_size.y =>
            {
                Some((voxel.x as u32, voxel.y as u32, (-1 - voxel.z) as u32))
            }
            Self::PosZ
                if voxel.z >= field_size.z
                    && voxel.x >= 0
                    && voxel.y >= 0
                    && voxel.x < field_size.x
                    && voxel.y < field_size.y =>
            {
                Some((
                    voxel.x as u32,
                    voxel.y as u32,
                    (voxel.z - field_size.z) as u32,
                ))
            }
            _ => None,
        }
    }
}

/// A 2D slice of neighbor data with depth planes.
#[derive(Clone, Debug)]
pub struct NeighborSlice<T> {
    pub data: Vec<T>,
    pub size_a: u32,
    pub size_b: u32,
    pub depth: u32,
}

impl<T: Copy + Default> NeighborSlice<T> {
    /// Creates a slice by sampling with a given function.
    pub fn from_sampler<F>(face: NeighborFace, field_size: UVec3, sampler: F) -> Self
    where
        F: Fn(u32, u32, u32) -> T,
    {
        let (size_a, size_b) = face.slice_dimensions(field_size);
        let mut data = Vec::with_capacity((size_a * size_b * NEIGHBOR_DEPTH) as usize);

        for depth in 0..NEIGHBOR_DEPTH {
            for b in 0..size_b {
                for a in 0..size_a {
                    data.push(sampler(a, b, depth));
                }
            }
        }

        Self {
            data,
            size_a,
            size_b,
            depth: NEIGHBOR_DEPTH,
        }
    }

    /// Creates a slice from a field, extracting raw values.
    pub fn from_field<F: Field<T> + ?Sized>(field: &F, face: NeighborFace) -> Self {
        Self::from_sampler(face, F::SIZE, |a, b, depth| {
            let (x, y, z) = face.to_field_coords(a, b, depth, F::SIZE);
            field.get(x, y, z)
        })
    }

    /// Gets the value at (a, b, depth) coordinates.
    #[inline]
    pub fn get(&self, a: u32, b: u32, depth: u32) -> T {
        if a < self.size_a && b < self.size_b && depth < self.depth {
            let idx = (a + b * self.size_a + depth * self.size_a * self.size_b) as usize;
            self.data[idx]
        } else {
            T::default()
        }
    }
}

// ============================================================================
// Generic neighbor data for meshing
// ============================================================================

/// Cached neighbor values for seamless meshing.
///
/// Stores raw field values from neighboring chunks. The `Sculptable::to_iso`
/// conversion happens during mesh generation, allowing this to work with
/// any field type (f32 SDF, bool voxels, u8 materials, etc.).
///
/// # Usage
///
/// This component is automatically added to entities with `GenerateMesh` by
/// the registered sculptable systems. You typically don't create this manually.
#[derive(Component, Clone, Debug)]
pub struct NeighborFields<T: Copy + Default + Send + Sync + 'static> {
    pub neighbors: [Option<NeighborSlice<T>>; 6],
}

impl<T: Copy + Default + Send + Sync + 'static> Default for NeighborFields<T> {
    fn default() -> Self {
        Self {
            neighbors: Default::default(),
        }
    }
}

impl<T: Copy + Default + Send + Sync + 'static> NeighborFields<T> {
    /// Sample a raw value at the given voxel coordinate.
    ///
    /// Returns `Some(value)` if the voxel is in a neighbor's region and data exists.
    #[inline]
    pub fn sample(&self, voxel: IVec3, field_size: IVec3) -> Option<T> {
        for face in NeighborFace::ALL {
            if let Some((a, b, depth)) = face.voxel_to_slice_coords(voxel, field_size) {
                if let Some(ref slice) = self.neighbors[face as usize] {
                    return Some(slice.get(a, b, depth));
                }
            }
        }
        None
    }

    /// Sample using the field size from a Field type.
    #[inline]
    pub fn sample_for<F: Field<T>>(&self, voxel: IVec3) -> Option<T> {
        self.sample(voxel, F::SIZE.as_ivec3())
    }

    /// Check if neighbor data exists for a given face.
    pub fn has_neighbor(&self, face: NeighborFace) -> bool {
        self.neighbors[face as usize].is_some()
    }

    /// Count how many neighbors have data.
    pub fn neighbor_count(&self) -> usize {
        self.neighbors.iter().filter(|n| n.is_some()).count()
    }
}

// ============================================================================
// Backward compatibility
// ============================================================================

/// Backward compatibility alias for f32 neighbor fields.
#[deprecated(since = "0.19.0", note = "Renamed to NeighborFields<f32>")]
pub type NeighborIsoFields = NeighborFields<f32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_to_slice_coords() {
        let size = ivec3(32, 32, 32);

        assert_eq!(
            NeighborFace::NegX.voxel_to_slice_coords(ivec3(-1, 5, 10), size),
            Some((5, 10, 0))
        );
        assert_eq!(
            NeighborFace::PosX.voxel_to_slice_coords(ivec3(32, 5, 10), size),
            Some((5, 10, 0))
        );
        assert_eq!(
            NeighborFace::NegX.voxel_to_slice_coords(ivec3(5, 5, 5), size),
            None
        );
    }

    #[test]
    fn test_neighbor_fields_sample() {
        let mut fields: NeighborFields<f32> = NeighborFields::default();

        fields.neighbors[NeighborFace::PosX as usize] = Some(NeighborSlice::from_sampler(
            NeighborFace::PosX,
            uvec3(32, 32, 32),
            |_, _, _| -0.5,
        ));

        assert_eq!(
            fields.sample(ivec3(32, 5, 5), ivec3(32, 32, 32)),
            Some(-0.5)
        );
        assert_eq!(fields.sample(ivec3(5, 5, 5), ivec3(32, 32, 32)), None);
    }

    #[test]
    fn test_neighbor_fields_bool() {
        let mut fields: NeighborFields<bool> = NeighborFields::default();

        fields.neighbors[NeighborFace::PosX as usize] = Some(NeighborSlice::from_sampler(
            NeighborFace::PosX,
            uvec3(32, 32, 32),
            |_, _, _| true,
        ));

        assert_eq!(
            fields.sample(ivec3(32, 5, 5), ivec3(32, 32, 32)),
            Some(true)
        );
        assert_eq!(fields.sample(ivec3(5, 5, 5), ivec3(32, 32, 32)), None);
    }
}
