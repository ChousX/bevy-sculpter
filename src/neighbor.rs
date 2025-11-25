use bevy::prelude::*;

use crate::{DENSITY_FIELD_SIZE, density_field::DensityField};

/// Cached neighbor field data for seamless meshing
#[derive(Component, Clone, Debug, Default)]
pub struct NeighborDensityFields {
    pub neighbors: [Option<NeighborSlice>; 6],
}

#[derive(Clone, Debug)]
pub struct NeighborSlice {
    pub data: Vec<f32>,
    pub size_a: u32,
    pub size_b: u32,
}

impl NeighborSlice {
    /// Create slice from neighbor's boundary
    /// For NegX neighbor: we sample their x=MAX-1 face
    /// For PosX neighbor: we sample their x=0 face
    pub fn from_field(field: &DensityField, face: NeighborFace) -> Self {
        let (size_a, size_b, sampler): (u32, u32, Box<dyn Fn(u32, u32) -> f32>) = match face {
            // When we need data from -X neighbor, get their x=MAX-1 plane
            NeighborFace::NegX => (
                DENSITY_FIELD_SIZE.y,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a, b| field.get(DENSITY_FIELD_SIZE.x - 1, a, b)),
            ),
            // When we need data from +X neighbor, get their x=0 plane
            NeighborFace::PosX => (
                DENSITY_FIELD_SIZE.y,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a, b| field.get(0, a, b)),
            ),
            // When we need data from -Y neighbor, get their y=MAX-1 plane
            NeighborFace::NegY => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a, b| field.get(a, DENSITY_FIELD_SIZE.y - 1, b)),
            ),
            // When we need data from +Y neighbor, get their y=0 plane
            NeighborFace::PosY => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a, b| field.get(a, 0, b)),
            ),
            // When we need data from -Z neighbor, get their z=MAX-1 plane
            NeighborFace::NegZ => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.y,
                Box::new(|a, b| field.get(a, b, DENSITY_FIELD_SIZE.z - 1)),
            ),
            // When we need data from +Z neighbor, get their z=0 plane
            NeighborFace::PosZ => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.y,
                Box::new(|a, b| field.get(a, b, 0)),
            ),
        };

        let mut data = Vec::with_capacity((size_a * size_b) as usize);
        for b in 0..size_b {
            for a in 0..size_a {
                data.push(sampler(a, b));
            }
        }

        Self {
            data,
            size_a,
            size_b,
        }
    }

    #[inline]
    pub fn get(&self, a: u32, b: u32) -> f32 {
        if a < self.size_a && b < self.size_b {
            self.data[(a + b * self.size_a) as usize]
        } else {
            1.0 // Outside
        }
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
