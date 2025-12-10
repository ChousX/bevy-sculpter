//! Neighbor chunk data for seamless mesh boundaries.
//!
//! When meshing a chunk, the Surface Nets algorithm needs to sample density values
//! slightly beyond the chunk boundaries to properly connect vertices at the edges.
//! This module provides generic structures to cache and access that neighbor data.

use bevy::prelude::*;

use crate::{DENSITY_FIELD_SIZE, density_field::DensityField};

/// How many planes of neighbor data to store.
///
/// We need 2 planes for proper boundary vertex computation in Surface Nets.
pub const NEIGHBOR_DEPTH: u32 = 2;

/// Identifies a face of a chunk for neighbor lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeighborFace {
    /// Negative X direction (-1, 0, 0)
    NegX = 0,
    /// Positive X direction (+1, 0, 0)
    PosX = 1,
    /// Negative Y direction (0, -1, 0)
    NegY = 2,
    /// Positive Y direction (0, +1, 0)
    PosY = 3,
    /// Negative Z direction (0, 0, -1)
    NegZ = 4,
    /// Positive Z direction (0, 0, +1)
    PosZ = 5,
}

impl NeighborFace {
    /// All six faces in order matching the enum discriminants.
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
    ///
    /// For negative faces, depth=0 is at SIZE-1, depth=1 is at SIZE-2, etc.
    /// For positive faces, depth=0 is at 0, depth=1 is at 1, etc.
    #[inline]
    pub fn to_field_coords(&self, a: u32, b: u32, depth: u32, field_size: UVec3) -> (u32, u32, u32) {
        match self {
            Self::NegX => (field_size.x.saturating_sub(1 + depth), a, b),
            Self::PosX => (depth.min(field_size.x - 1), a, b),
            Self::NegY => (a, field_size.y.saturating_sub(1 + depth), b),
            Self::PosY => (a, depth.min(field_size.y - 1), b),
            Self::NegZ => (a, b, field_size.z.saturating_sub(1 + depth)),
            Self::PosZ => (a, b, depth.min(field_size.z - 1)),
        }
    }

    /// Check if a voxel coordinate is in this neighbor's territory and return slice coords.
    ///
    /// Returns `Some((a, b, depth))` if the voxel is in this neighbor's region,
    /// `None` otherwise.
    #[inline]
    pub fn voxel_to_slice_coords(&self, voxel: IVec3, field_size: IVec3) -> Option<(u32, u32, u32)> {
        match self {
            Self::NegX if voxel.x < 0 
                && voxel.y >= 0 && voxel.z >= 0 
                && voxel.y < field_size.y && voxel.z < field_size.z => {
                Some((voxel.y as u32, voxel.z as u32, (-1 - voxel.x) as u32))
            }
            Self::PosX if voxel.x >= field_size.x 
                && voxel.y >= 0 && voxel.z >= 0 
                && voxel.y < field_size.y && voxel.z < field_size.z => {
                Some((voxel.y as u32, voxel.z as u32, (voxel.x - field_size.x) as u32))
            }
            Self::NegY if voxel.y < 0 
                && voxel.x >= 0 && voxel.z >= 0 
                && voxel.x < field_size.x && voxel.z < field_size.z => {
                Some((voxel.x as u32, voxel.z as u32, (-1 - voxel.y) as u32))
            }
            Self::PosY if voxel.y >= field_size.y 
                && voxel.x >= 0 && voxel.z >= 0 
                && voxel.x < field_size.x && voxel.z < field_size.z => {
                Some((voxel.x as u32, voxel.z as u32, (voxel.y - field_size.y) as u32))
            }
            Self::NegZ if voxel.z < 0 
                && voxel.x >= 0 && voxel.y >= 0 
                && voxel.x < field_size.x && voxel.y < field_size.y => {
                Some((voxel.x as u32, voxel.y as u32, (-1 - voxel.z) as u32))
            }
            Self::PosZ if voxel.z >= field_size.z 
                && voxel.x >= 0 && voxel.y >= 0 
                && voxel.x < field_size.x && voxel.y < field_size.y => {
                Some((voxel.x as u32, voxel.y as u32, (voxel.z - field_size.z) as u32))
            }
            _ => None,
        }
    }
}

/// Generic neighbor slice that stores boundary planes of data from a neighboring chunk.
///
/// Contains [`NEIGHBOR_DEPTH`] planes of data (typically 2) to allow proper
/// gradient computation and vertex positioning at chunk boundaries.
///
/// This is generic over the element type `T`, allowing reuse for both
/// density fields (f32) and material fields (u8).
#[derive(Clone, Debug)]
pub struct NeighborSlice<T> {
    /// Flattened 3D data: `[depth][b][a]` stored as `[a + b * size_a + depth * size_a * size_b]`
    pub data: Vec<T>,
    /// Size along the first axis (depends on face orientation).
    pub size_a: u32,
    /// Size along the second axis (depends on face orientation).
    pub size_b: u32,
    /// Number of planes stored (typically [`NEIGHBOR_DEPTH`]).
    pub depth: u32,
}

impl<T: Copy + Default> NeighborSlice<T> {
    /// Creates a slice by sampling with a given function.
    ///
    /// The sampler receives `(a, b, depth)` coordinates and returns the value.
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

    /// Gets the value at (a, b) coordinates with depth offset.
    ///
    /// # Arguments
    /// * `a` - First axis coordinate
    /// * `b` - Second axis coordinate  
    /// * `depth` - Depth into the neighbor (0 = closest plane, 1 = one step further)
    ///
    /// # Returns
    /// The value, or `T::default()` if out of bounds.
    #[inline]
    pub fn get(&self, a: u32, b: u32, depth: u32) -> T {
        if a < self.size_a && b < self.size_b && depth < self.depth {
            let idx = (a + b * self.size_a + depth * self.size_a * self.size_b) as usize;
            self.data[idx]
        } else {
            T::default()
        }
    }

    /// Gets the value at the boundary plane (depth=0).
    ///
    /// Convenience method equivalent to `get(a, b, 0)`.
    #[inline]
    pub fn get_boundary(&self, a: u32, b: u32) -> T {
        self.get(a, b, 0)
    }
}

/// Generic cached neighbor field data for seamless meshing.
///
/// Stores boundary slices from up to 6 neighboring chunks (one per face).
/// This allows the mesher to sample values beyond chunk boundaries
/// without querying the full neighbor fields.
#[derive(Component, Clone, Debug, Default)]
pub struct NeighborFields<T: Clone + Default + Send + Sync + 'static> {
    /// Neighbor slices indexed by [`NeighborFace`] (0=NegX, 1=PosX, 2=NegY, etc.)
    pub neighbors: [Option<NeighborSlice<T>>; 6],
}

impl<T: Copy + Default + Send + Sync + 'static> NeighborFields<T> {
    /// Sample a value from neighbors at the given voxel coordinate.
    ///
    /// Returns `Some(value)` if the voxel is in a neighbor's region and that
    /// neighbor data exists, `None` otherwise.
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
}

// ============================================================================
// Type aliases for convenience
// ============================================================================

/// Neighbor slice for density field data (f32).
pub type DensitySlice = NeighborSlice<f32>;

/// Cached neighbor density data for seamless meshing.
pub type NeighborDensityFields = NeighborFields<f32>;

// ============================================================================
// DensityField-specific implementation
// ============================================================================

impl NeighborSlice<f32> {
    /// Creates a density slice from a neighbor chunk's boundary planes.
    ///
    /// # Arguments
    /// * `field` - The neighbor's density field
    /// * `face` - Which face of the neighbor to sample (from the current chunk's perspective)
    pub fn from_density_field(field: &DensityField, face: NeighborFace) -> Self {
        Self::from_sampler(face, DENSITY_FIELD_SIZE, |a, b, depth| {
            let (x, y, z) = face.to_field_coords(a, b, depth, DENSITY_FIELD_SIZE);
            field.get(x, y, z)
        })
    }
}

// Keep the old name working for backwards compatibility
impl NeighborSlice<f32> {
    /// Creates a slice from a neighbor chunk's boundary planes.
    ///
    /// Deprecated: Use `from_density_field` instead.
    #[deprecated(since = "0.2.0", note = "Use from_density_field instead")]
    pub fn from_field(field: &DensityField, face: NeighborFace) -> Self {
        Self::from_density_field(field, face)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_dimensions() {
        let size = uvec3(32, 32, 32);
        assert_eq!(NeighborFace::NegX.slice_dimensions(size), (32, 32));
        assert_eq!(NeighborFace::PosY.slice_dimensions(size), (32, 32));
    }

    #[test]
    fn test_voxel_to_slice_coords() {
        let size = ivec3(32, 32, 32);
        
        // Test NegX: voxel at x=-1 should map to depth=0
        assert_eq!(
            NeighborFace::NegX.voxel_to_slice_coords(ivec3(-1, 5, 10), size),
            Some((5, 10, 0))
        );
        
        // Test PosX: voxel at x=32 should map to depth=0
        assert_eq!(
            NeighborFace::PosX.voxel_to_slice_coords(ivec3(32, 5, 10), size),
            Some((5, 10, 0))
        );
        
        // Test in-bounds returns None
        assert_eq!(
            NeighborFace::NegX.voxel_to_slice_coords(ivec3(5, 5, 5), size),
            None
        );
    }

    #[test]
    fn test_generic_slice() {
        // Test with u8
        let slice: NeighborSlice<u8> = NeighborSlice::from_sampler(
            NeighborFace::PosX,
            uvec3(32, 32, 32),
            |a, b, _depth| ((a + b) % 256) as u8,
        );
        
        assert_eq!(slice.get(0, 0, 0), 0);
        assert_eq!(slice.get(1, 1, 0), 2);
        assert_eq!(slice.get(100, 100, 0), 0); // Out of bounds returns default
    }

    #[test]
    fn test_neighbor_fields_sample() {
        let mut fields: NeighborFields<u8> = NeighborFields::default();
        
        // Add a slice for PosX neighbor
        fields.neighbors[NeighborFace::PosX as usize] = Some(NeighborSlice::from_sampler(
            NeighborFace::PosX,
            uvec3(32, 32, 32),
            |_a, _b, _depth| 42u8,
        ));
        
        let size = ivec3(32, 32, 32);
        
        // Sample from PosX neighbor region
        assert_eq!(fields.sample(ivec3(32, 5, 5), size), Some(42));
        
        // Sample from in-bounds (no neighbor)
        assert_eq!(fields.sample(ivec3(5, 5, 5), size), None);
        
        // Sample from NegX neighbor region (no data)
        assert_eq!(fields.sample(ivec3(-1, 5, 5), size), None);
    }
}
