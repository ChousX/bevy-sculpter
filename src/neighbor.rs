use bevy::prelude::*;

use crate::{DENSITY_FIELD_SIZE, density_field::DensityField};

/// How many planes of neighbor data to store (need 2 for proper boundary vertex computation)
pub const NEIGHBOR_DEPTH: u32 = 2;

/// Cached neighbor field data for seamless meshing
#[derive(Component, Clone, Debug, Default)]
pub struct NeighborDensityFields {
    pub neighbors: [Option<NeighborSlice>; 6],
}

/// Stores 2 planes of neighbor density data for seamless boundary meshing
#[derive(Clone, Debug)]
pub struct NeighborSlice {
    /// Flattened 3D data: [depth][b][a] stored as [a + b * size_a + depth * size_a * size_b]
    pub data: Vec<f32>,
    pub size_a: u32,
    pub size_b: u32,
    pub depth: u32,
}

impl NeighborSlice {
    /// Create slice from neighbor's boundary (2 planes deep)
    /// For PosX neighbor: we sample their x=0 and x=1 faces
    /// For NegX neighbor: we sample their x=SIZE-1 and x=SIZE-2 faces
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

    /// Get value at (a, b) coordinates with depth offset
    /// depth=0 is the plane closest to the boundary
    /// depth=1 is one step further into the neighbor
    #[inline]
    pub fn get(&self, a: u32, b: u32, depth: u32) -> f32 {
        if a < self.size_a && b < self.size_b && depth < self.depth {
            let idx = (a + b * self.size_a + depth * self.size_a * self.size_b) as usize;
            self.data[idx]
        } else {
            1.0 // Outside
        }
    }

    /// Convenience method for backward compatibility - gets depth=0
    #[inline]
    pub fn get_boundary(&self, a: u32, b: u32) -> f32 {
        self.get(a, b, 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
