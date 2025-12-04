//! Neighbor chunk data for seamless mesh boundaries.
//!
//! When meshing a chunk, the Surface Nets algorithm needs to sample density values
//! slightly beyond the chunk boundaries to properly connect vertices at the edges.
//! This module provides structures to cache and access that neighbor data.

use bevy::prelude::*;

use crate::{DENSITY_FIELD_SIZE, density_field::DensityField};

/// How many planes of neighbor data to store.
///
/// We need 2 planes for proper boundary vertex computation in Surface Nets.
pub const NEIGHBOR_DEPTH: u32 = 2;

/// Cached neighbor field data for seamless meshing.
///
/// Stores boundary slices from up to 6 neighboring chunks (one per face).
/// This allows the mesher to sample density values beyond chunk boundaries
/// without querying the full neighbor fields.
///
/// # Example
///
/// ```ignore
/// // The SurfaceNetsPlugin automatically gathers this data, but you can
/// // also construct it manually:
/// let mut neighbors = NeighborDensityFields::default();
/// neighbors.neighbors[NeighborFace::PosX as usize] =
///     Some(NeighborSlice::from_field(&neighbor_field, NeighborFace::PosX));
/// ```
#[derive(Component, Clone, Debug, Default)]
pub struct NeighborDensityFields {
    /// Neighbor slices indexed by [`NeighborFace`] (0=NegX, 1=PosX, 2=NegY, etc.)
    pub neighbors: [Option<NeighborSlice>; 6],
}

/// Stores boundary planes of density data from a neighboring chunk.
///
/// Contains [`NEIGHBOR_DEPTH`] planes of data (typically 2) to allow proper
/// gradient computation and vertex positioning at chunk boundaries.
#[derive(Clone, Debug)]
pub struct NeighborSlice {
    /// Flattened 3D data: `[depth][b][a]` stored as `[a + b * size_a + depth * size_a * size_b]`
    pub data: Vec<f32>,
    /// Size along the first axis (depends on face orientation).
    pub size_a: u32,
    /// Size along the second axis (depends on face orientation).
    pub size_b: u32,
    /// Number of planes stored (typically [`NEIGHBOR_DEPTH`]).
    pub depth: u32,
}

impl NeighborSlice {
    /// Creates a slice from a neighbor chunk's boundary planes.
    ///
    /// Extracts [`NEIGHBOR_DEPTH`] planes of data from the appropriate face of the neighbor.
    ///
    /// # Arguments
    /// * `field` - The neighbor's density field
    /// * `face` - Which face of the neighbor to sample (from the current chunk's perspective)
    ///
    /// # Face Mapping
    /// - `PosX` neighbor: samples their x=0 and x=1 faces (corresponds to our x=SIZE and x=SIZE+1)
    /// - `NegX` neighbor: samples their x=SIZE-1 and x=SIZE-2 faces (corresponds to our x=-1 and x=-2)
    /// - Similar logic for Y and Z axes
    pub fn from_field(field: &DensityField, face: NeighborFace) -> Self {
        let (size_a, size_b, sampler): (u32, u32, Box<dyn Fn(u32, u32, u32) -> f32>) = match face {
            // When we need data from -X neighbor, get their x=SIZE-1 and x=SIZE-2 planes
            // depth=0 -> x=SIZE-1 (corresponds to our x=-1)
            // depth=1 -> x=SIZE-2 (corresponds to our x=-2)
            NeighborFace::NegX => (
                DENSITY_FIELD_SIZE.y,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a, b, depth| {
                    let x = DENSITY_FIELD_SIZE.x.saturating_sub(1 + depth);
                    field.get(x, a, b)
                }),
            ),
            // When we need data from +X neighbor, get their x=0 and x=1 planes
            // depth=0 -> x=0 (corresponds to our x=SIZE)
            // depth=1 -> x=1 (corresponds to our x=SIZE+1)
            NeighborFace::PosX => (
                DENSITY_FIELD_SIZE.y,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a, b, depth| field.get(depth.min(DENSITY_FIELD_SIZE.x - 1), a, b)),
            ),
            // When we need data from -Y neighbor, get their y=SIZE-1 and y=SIZE-2 planes
            NeighborFace::NegY => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a, b, depth| {
                    let y = DENSITY_FIELD_SIZE.y.saturating_sub(1 + depth);
                    field.get(a, y, b)
                }),
            ),
            // When we need data from +Y neighbor, get their y=0 and y=1 planes
            NeighborFace::PosY => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a, b, depth| field.get(a, depth.min(DENSITY_FIELD_SIZE.y - 1), b)),
            ),
            // When we need data from -Z neighbor, get their z=SIZE-1 and z=SIZE-2 planes
            NeighborFace::NegZ => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.y,
                Box::new(|a, b, depth| {
                    let z = DENSITY_FIELD_SIZE.z.saturating_sub(1 + depth);
                    field.get(a, b, z)
                }),
            ),
            // When we need data from +Z neighbor, get their z=0 and z=1 planes
            NeighborFace::PosZ => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.y,
                Box::new(|a, b, depth| field.get(a, b, depth.min(DENSITY_FIELD_SIZE.z - 1))),
            ),
        };

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

    /// Gets the density value at (a, b) coordinates with depth offset.
    ///
    /// # Arguments
    /// * `a` - First axis coordinate
    /// * `b` - Second axis coordinate  
    /// * `depth` - Depth into the neighbor (0 = closest plane, 1 = one step further)
    ///
    /// # Returns
    /// The density value, or `1.0` (exterior) if out of bounds.
    #[inline]
    pub fn get(&self, a: u32, b: u32, depth: u32) -> f32 {
        if a < self.size_a && b < self.size_b && depth < self.depth {
            let idx = (a + b * self.size_a + depth * self.size_a * self.size_b) as usize;
            self.data[idx]
        } else {
            1.0 // Outside
        }
    }

    /// Gets the density value at the boundary plane (depth=0).
    ///
    /// Convenience method equivalent to `get(a, b, 0)`.
    #[inline]
    pub fn get_boundary(&self, a: u32, b: u32) -> f32 {
        self.get(a, b, 0)
    }
}

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
}
