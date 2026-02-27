//! Neighbor chunk data for seamless mesh boundaries.
//!
//! When meshing a chunk, the Surface Nets algorithm needs to sample density values
//! slightly beyond the chunk boundaries to properly connect vertices at the edges.
//! This module provides structures to cache and access that neighbor data.
//!
//! Supports all 26 neighbors: 6 faces, 12 edges, and 8 corners.

use crate::prelude::*;
use bevy::prelude::*;

/// Default neighbor depth when no LOD is specified.
pub const NEIGHBOR_DEPTH: u32 = 2;

/// Compute the neighbor depth needed for a given LOD step.
///
/// The mesher samples corners at `pos + step`, so we need `step + 1`
/// planes of neighbor data to cover all possible samples.
#[inline]
pub fn neighbor_depth_for_step(step: u32) -> u32 {
    step + 1
}

// Face neighbors (6)
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
    /// Only matches when exactly one axis is out of bounds (face region).
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

// Edge neighbors (12)
/// Identifies an edge of a chunk (where two faces meet).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeighborEdge {
    /// -X, -Y edge (runs along Z)
    NegXNegY = 0,
    /// -X, +Y edge (runs along Z)
    NegXPosY = 1,
    /// +X, -Y edge (runs along Z)
    PosXNegY = 2,
    /// +X, +Y edge (runs along Z)
    PosXPosY = 3,
    /// -X, -Z edge (runs along Y)
    NegXNegZ = 4,
    /// -X, +Z edge (runs along Y)
    NegXPosZ = 5,
    /// +X, -Z edge (runs along Y)
    PosXNegZ = 6,
    /// +X, +Z edge (runs along Y)
    PosXPosZ = 7,
    /// -Y, -Z edge (runs along X)
    NegYNegZ = 8,
    /// -Y, +Z edge (runs along X)
    NegYPosZ = 9,
    /// +Y, -Z edge (runs along X)
    PosYNegZ = 10,
    /// +Y, +Z edge (runs along X)
    PosYPosZ = 11,
}

impl NeighborEdge {
    pub const ALL: [Self; 12] = [
        Self::NegXNegY,
        Self::NegXPosY,
        Self::PosXNegY,
        Self::PosXPosY,
        Self::NegXNegZ,
        Self::NegXPosZ,
        Self::PosXNegZ,
        Self::PosXPosZ,
        Self::NegYNegZ,
        Self::NegYPosZ,
        Self::PosYNegZ,
        Self::PosYPosZ,
    ];

    /// Returns the chunk coordinate offset for this edge direction.
    pub fn offset(&self) -> IVec3 {
        match self {
            Self::NegXNegY => ivec3(-1, -1, 0),
            Self::NegXPosY => ivec3(-1, 1, 0),
            Self::PosXNegY => ivec3(1, -1, 0),
            Self::PosXPosY => ivec3(1, 1, 0),
            Self::NegXNegZ => ivec3(-1, 0, -1),
            Self::NegXPosZ => ivec3(-1, 0, 1),
            Self::PosXNegZ => ivec3(1, 0, -1),
            Self::PosXPosZ => ivec3(1, 0, 1),
            Self::NegYNegZ => ivec3(0, -1, -1),
            Self::NegYPosZ => ivec3(0, -1, 1),
            Self::PosYNegZ => ivec3(0, 1, -1),
            Self::PosYPosZ => ivec3(0, 1, 1),
        }
    }

    /// Returns the length of this edge's free axis.
    #[inline]
    pub fn axis_length(&self, field_size: UVec3) -> u32 {
        match self {
            Self::NegXNegY | Self::NegXPosY | Self::PosXNegY | Self::PosXPosY => field_size.z,
            Self::NegXNegZ | Self::NegXPosZ | Self::PosXNegZ | Self::PosXPosZ => field_size.y,
            Self::NegYNegZ | Self::NegYPosZ | Self::PosYNegZ | Self::PosYPosZ => field_size.x,
        }
    }

    /// Convert edge-local coordinates to field coordinates for sampling the neighbor.
    ///
    /// `a` is along the free axis, `depth_u`/`depth_v` are depths into each OOB axis.
    #[inline]
    pub fn to_field_coords(
        &self,
        a: u32,
        depth_u: u32,
        depth_v: u32,
        field_size: UVec3,
    ) -> (u32, u32, u32) {
        let s = field_size;
        match self {
            // Runs along Z: OOB axes are X and Y
            Self::NegXNegY => (
                s.x.saturating_sub(1 + depth_u),
                s.y.saturating_sub(1 + depth_v),
                a,
            ),
            Self::NegXPosY => (s.x.saturating_sub(1 + depth_u), depth_v.min(s.y - 1), a),
            Self::PosXNegY => (depth_u.min(s.x - 1), s.y.saturating_sub(1 + depth_v), a),
            Self::PosXPosY => (depth_u.min(s.x - 1), depth_v.min(s.y - 1), a),
            // Runs along Y: OOB axes are X and Z
            Self::NegXNegZ => (
                s.x.saturating_sub(1 + depth_u),
                a,
                s.z.saturating_sub(1 + depth_v),
            ),
            Self::NegXPosZ => (s.x.saturating_sub(1 + depth_u), a, depth_v.min(s.z - 1)),
            Self::PosXNegZ => (depth_u.min(s.x - 1), a, s.z.saturating_sub(1 + depth_v)),
            Self::PosXPosZ => (depth_u.min(s.x - 1), a, depth_v.min(s.z - 1)),
            // Runs along X: OOB axes are Y and Z
            Self::NegYNegZ => (
                a,
                s.y.saturating_sub(1 + depth_u),
                s.z.saturating_sub(1 + depth_v),
            ),
            Self::NegYPosZ => (a, s.y.saturating_sub(1 + depth_u), depth_v.min(s.z - 1)),
            Self::PosYNegZ => (a, depth_u.min(s.y - 1), s.z.saturating_sub(1 + depth_v)),
            Self::PosYPosZ => (a, depth_u.min(s.y - 1), depth_v.min(s.z - 1)),
        }
    }

    /// Check if a voxel coordinate is in this edge neighbor's territory.
    ///
    /// Returns `Some((a, depth_u, depth_v))` if the voxel is in this edge's region
    /// (exactly 2 axes out of bounds).
    #[inline]
    pub fn voxel_to_edge_coords(&self, voxel: IVec3, field_size: IVec3) -> Option<(u32, u32, u32)> {
        let s = field_size;
        match self {
            Self::NegXNegY if voxel.x < 0 && voxel.y < 0 && voxel.z >= 0 && voxel.z < s.z => {
                Some((voxel.z as u32, (-1 - voxel.x) as u32, (-1 - voxel.y) as u32))
            }
            Self::NegXPosY if voxel.x < 0 && voxel.y >= s.y && voxel.z >= 0 && voxel.z < s.z => {
                Some((
                    voxel.z as u32,
                    (-1 - voxel.x) as u32,
                    (voxel.y - s.y) as u32,
                ))
            }
            Self::PosXNegY if voxel.x >= s.x && voxel.y < 0 && voxel.z >= 0 && voxel.z < s.z => {
                Some((
                    voxel.z as u32,
                    (voxel.x - s.x) as u32,
                    (-1 - voxel.y) as u32,
                ))
            }
            Self::PosXPosY if voxel.x >= s.x && voxel.y >= s.y && voxel.z >= 0 && voxel.z < s.z => {
                Some((
                    voxel.z as u32,
                    (voxel.x - s.x) as u32,
                    (voxel.y - s.y) as u32,
                ))
            }
            Self::NegXNegZ if voxel.x < 0 && voxel.z < 0 && voxel.y >= 0 && voxel.y < s.y => {
                Some((voxel.y as u32, (-1 - voxel.x) as u32, (-1 - voxel.z) as u32))
            }
            Self::NegXPosZ if voxel.x < 0 && voxel.z >= s.z && voxel.y >= 0 && voxel.y < s.y => {
                Some((
                    voxel.y as u32,
                    (-1 - voxel.x) as u32,
                    (voxel.z - s.z) as u32,
                ))
            }
            Self::PosXNegZ if voxel.x >= s.x && voxel.z < 0 && voxel.y >= 0 && voxel.y < s.y => {
                Some((
                    voxel.y as u32,
                    (voxel.x - s.x) as u32,
                    (-1 - voxel.z) as u32,
                ))
            }
            Self::PosXPosZ if voxel.x >= s.x && voxel.z >= s.z && voxel.y >= 0 && voxel.y < s.y => {
                Some((
                    voxel.y as u32,
                    (voxel.x - s.x) as u32,
                    (voxel.z - s.z) as u32,
                ))
            }
            Self::NegYNegZ if voxel.y < 0 && voxel.z < 0 && voxel.x >= 0 && voxel.x < s.x => {
                Some((voxel.x as u32, (-1 - voxel.y) as u32, (-1 - voxel.z) as u32))
            }
            Self::NegYPosZ if voxel.y < 0 && voxel.z >= s.z && voxel.x >= 0 && voxel.x < s.x => {
                Some((
                    voxel.x as u32,
                    (-1 - voxel.y) as u32,
                    (voxel.z - s.z) as u32,
                ))
            }
            Self::PosYNegZ if voxel.y >= s.y && voxel.z < 0 && voxel.x >= 0 && voxel.x < s.x => {
                Some((
                    voxel.x as u32,
                    (voxel.y - s.y) as u32,
                    (-1 - voxel.z) as u32,
                ))
            }
            Self::PosYPosZ if voxel.y >= s.y && voxel.z >= s.z && voxel.x >= 0 && voxel.x < s.x => {
                Some((
                    voxel.x as u32,
                    (voxel.y - s.y) as u32,
                    (voxel.z - s.z) as u32,
                ))
            }
            _ => None,
        }
    }
}

// ============================================================================
// Corner neighbors (8)
// ============================================================================

/// Identifies a corner of a chunk (where three faces meet).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeighborCorner {
    NegXNegYNegZ = 0,
    PosXNegYNegZ = 1,
    NegXPosYNegZ = 2,
    PosXPosYNegZ = 3,
    NegXNegYPosZ = 4,
    PosXNegYPosZ = 5,
    NegXPosYPosZ = 6,
    PosXPosYPosZ = 7,
}

impl NeighborCorner {
    pub const ALL: [Self; 8] = [
        Self::NegXNegYNegZ,
        Self::PosXNegYNegZ,
        Self::NegXPosYNegZ,
        Self::PosXPosYNegZ,
        Self::NegXNegYPosZ,
        Self::PosXNegYPosZ,
        Self::NegXPosYPosZ,
        Self::PosXPosYPosZ,
    ];

    /// Returns the chunk coordinate offset for this corner.
    pub fn offset(&self) -> IVec3 {
        match self {
            Self::NegXNegYNegZ => ivec3(-1, -1, -1),
            Self::PosXNegYNegZ => ivec3(1, -1, -1),
            Self::NegXPosYNegZ => ivec3(-1, 1, -1),
            Self::PosXPosYNegZ => ivec3(1, 1, -1),
            Self::NegXNegYPosZ => ivec3(-1, -1, 1),
            Self::PosXNegYPosZ => ivec3(1, -1, 1),
            Self::NegXPosYPosZ => ivec3(-1, 1, 1),
            Self::PosXPosYPosZ => ivec3(1, 1, 1),
        }
    }

    /// Convert corner-local depth coordinates to field coordinates for sampling.
    #[inline]
    pub fn to_field_coords(
        &self,
        depth_x: u32,
        depth_y: u32,
        depth_z: u32,
        field_size: UVec3,
    ) -> (u32, u32, u32) {
        let s = field_size;
        let x = match self {
            Self::NegXNegYNegZ | Self::NegXPosYNegZ | Self::NegXNegYPosZ | Self::NegXPosYPosZ => {
                s.x.saturating_sub(1 + depth_x)
            }
            _ => depth_x.min(s.x - 1),
        };
        let y = match self {
            Self::NegXNegYNegZ | Self::PosXNegYNegZ | Self::NegXNegYPosZ | Self::PosXNegYPosZ => {
                s.y.saturating_sub(1 + depth_y)
            }
            _ => depth_y.min(s.y - 1),
        };
        let z = match self {
            Self::NegXNegYNegZ | Self::PosXNegYNegZ | Self::NegXPosYNegZ | Self::PosXPosYNegZ => {
                s.z.saturating_sub(1 + depth_z)
            }
            _ => depth_z.min(s.z - 1),
        };
        (x, y, z)
    }

    /// Check if a voxel coordinate is in this corner neighbor's territory.
    ///
    /// Returns `Some((depth_x, depth_y, depth_z))` if all 3 axes are out of bounds
    /// in the matching directions.
    #[inline]
    pub fn voxel_to_corner_coords(
        &self,
        voxel: IVec3,
        field_size: IVec3,
    ) -> Option<(u32, u32, u32)> {
        let s = field_size;
        match self {
            Self::NegXNegYNegZ if voxel.x < 0 && voxel.y < 0 && voxel.z < 0 => Some((
                (-1 - voxel.x) as u32,
                (-1 - voxel.y) as u32,
                (-1 - voxel.z) as u32,
            )),
            Self::PosXNegYNegZ if voxel.x >= s.x && voxel.y < 0 && voxel.z < 0 => Some((
                (voxel.x - s.x) as u32,
                (-1 - voxel.y) as u32,
                (-1 - voxel.z) as u32,
            )),
            Self::NegXPosYNegZ if voxel.x < 0 && voxel.y >= s.y && voxel.z < 0 => Some((
                (-1 - voxel.x) as u32,
                (voxel.y - s.y) as u32,
                (-1 - voxel.z) as u32,
            )),
            Self::PosXPosYNegZ if voxel.x >= s.x && voxel.y >= s.y && voxel.z < 0 => Some((
                (voxel.x - s.x) as u32,
                (voxel.y - s.y) as u32,
                (-1 - voxel.z) as u32,
            )),
            Self::NegXNegYPosZ if voxel.x < 0 && voxel.y < 0 && voxel.z >= s.z => Some((
                (-1 - voxel.x) as u32,
                (-1 - voxel.y) as u32,
                (voxel.z - s.z) as u32,
            )),
            Self::PosXNegYPosZ if voxel.x >= s.x && voxel.y < 0 && voxel.z >= s.z => Some((
                (voxel.x - s.x) as u32,
                (-1 - voxel.y) as u32,
                (voxel.z - s.z) as u32,
            )),
            Self::NegXPosYPosZ if voxel.x < 0 && voxel.y >= s.y && voxel.z >= s.z => Some((
                (-1 - voxel.x) as u32,
                (voxel.y - s.y) as u32,
                (voxel.z - s.z) as u32,
            )),
            Self::PosXPosYPosZ if voxel.x >= s.x && voxel.y >= s.y && voxel.z >= s.z => Some((
                (voxel.x - s.x) as u32,
                (voxel.y - s.y) as u32,
                (voxel.z - s.z) as u32,
            )),
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
    pub fn from_sampler<F>(face: NeighborFace, field_size: UVec3, depth: u32, sampler: F) -> Self
    where
        F: Fn(u32, u32, u32) -> T,
    {
        let (size_a, size_b) = face.slice_dimensions(field_size);
        let mut data = Vec::with_capacity((size_a * size_b * depth) as usize);

        for d in 0..depth {
            for b in 0..size_b {
                for a in 0..size_a {
                    data.push(sampler(a, b, d));
                }
            }
        }

        Self {
            data,
            size_a,
            size_b,
            depth,
        }
    }

    /// Creates a slice from a field with the default depth.
    pub fn from_field<F: Field<T> + ?Sized>(field: &F, face: NeighborFace) -> Self {
        Self::from_field_with_depth(field, face, NEIGHBOR_DEPTH)
    }

    /// Creates a slice from a field with a specific depth.
    pub fn from_field_with_depth<F: Field<T> + ?Sized>(
        field: &F,
        face: NeighborFace,
        depth: u32,
    ) -> Self {
        Self::from_sampler(face, F::SIZE, depth, |a, b, d| {
            let (x, y, z) = face.to_field_coords(a, b, d, F::SIZE);
            field.get(x, y, z)
        })
    }

    /// Gets the value at (a, b, depth) coordinates.
    #[inline]
    pub fn get(&self, a: u32, b: u32, depth: u32) -> Option<T> {
        if a < self.size_a && b < self.size_b && depth < self.depth {
            let idx = (a + b * self.size_a + depth * self.size_a * self.size_b) as usize;
            Some(self.data[idx])
        } else {
            None
        }
    }
}

/// Edge neighbor data: a 1D strip with depth in two axes.
///
/// Layout: `data[a + depth_u * axis_len + depth_v * axis_len * depth]`
#[derive(Clone, Debug)]
pub struct NeighborEdgeSlice<T> {
    pub data: Vec<T>,
    pub axis_len: u32,
    pub depth: u32,
}

impl<T: Copy + Default> NeighborEdgeSlice<T> {
    /// Creates an edge slice from a field with the default depth.
    pub fn from_field<F: Field<T> + ?Sized>(field: &F, edge: NeighborEdge) -> Self {
        Self::from_field_with_depth(field, edge, NEIGHBOR_DEPTH)
    }

    /// Creates an edge slice from a field with a specific depth.
    pub fn from_field_with_depth<F: Field<T> + ?Sized>(
        field: &F,
        edge: NeighborEdge,
        depth: u32,
    ) -> Self {
        let axis_len = edge.axis_length(F::SIZE);
        let cap = (axis_len * depth * depth) as usize;
        let mut data = Vec::with_capacity(cap);

        for dv in 0..depth {
            for du in 0..depth {
                for a in 0..axis_len {
                    let (x, y, z) = edge.to_field_coords(a, du, dv, F::SIZE);
                    data.push(field.get(x, y, z));
                }
            }
        }

        Self {
            data,
            axis_len,
            depth,
        }
    }

    /// Gets the value at (a, depth_u, depth_v) coordinates.
    #[inline]
    pub fn get(&self, a: u32, depth_u: u32, depth_v: u32) -> Option<T> {
        if a < self.axis_len && depth_u < self.depth && depth_v < self.depth {
            let idx = (a + depth_u * self.axis_len + depth_v * self.axis_len * self.depth) as usize;
            Some(self.data[idx])
        } else {
            None
        }
    }
}

/// Corner neighbor data: a small depth³ cube.
#[derive(Clone, Debug)]
pub struct NeighborCornerSlice<T> {
    pub data: Vec<T>,
    pub depth: u32,
}

impl<T: Copy + Default> NeighborCornerSlice<T> {
    /// Creates a corner slice from a field with the default depth.
    pub fn from_field<F: Field<T> + ?Sized>(field: &F, corner: NeighborCorner) -> Self {
        Self::from_field_with_depth(field, corner, NEIGHBOR_DEPTH)
    }

    /// Creates a corner slice from a field with a specific depth.
    pub fn from_field_with_depth<F: Field<T> + ?Sized>(
        field: &F,
        corner: NeighborCorner,
        depth: u32,
    ) -> Self {
        let cap = (depth * depth * depth) as usize;
        let mut data = Vec::with_capacity(cap);

        for dz in 0..depth {
            for dy in 0..depth {
                for dx in 0..depth {
                    let (x, y, z) = corner.to_field_coords(dx, dy, dz, F::SIZE);
                    data.push(field.get(x, y, z));
                }
            }
        }

        Self { data, depth }
    }

    /// Gets the value at (depth_x, depth_y, depth_z) coordinates.
    #[inline]
    pub fn get(&self, dx: u32, dy: u32, dz: u32) -> Option<T> {
        if dx < self.depth && dy < self.depth && dz < self.depth {
            let idx = (dx + dy * self.depth + dz * self.depth * self.depth) as usize;
            Some(self.data[idx])
        } else {
            None
        }
    }
}

/// Cached neighbor values for seamless meshing.
///
/// Stores raw field values from all 26 neighboring chunks (6 faces, 12 edges,
/// 8 corners). The `Sculptable::to_iso` conversion happens during mesh
/// generation, allowing this to work with any field type.
#[derive(Component, Clone, Debug)]
pub struct NeighborFields<T: Copy + Default + Send + Sync + 'static> {
    pub neighbors: [Option<NeighborSlice<T>>; 6],
    pub edges: [Option<NeighborEdgeSlice<T>>; 12],
    pub corners: [Option<NeighborCornerSlice<T>>; 8],
}

impl<T: Copy + Default + Send + Sync + 'static> Default for NeighborFields<T> {
    fn default() -> Self {
        Self {
            neighbors: [const { None }; 6],
            edges: [const { None }; 12],
            corners: [const { None }; 8],
        }
    }
}

impl<T: Copy + Default + Send + Sync + 'static> NeighborFields<T> {
    /// Gather neighbor data using a closure that returns an optional field reference for each face.
    pub fn gather<'a, F, Fld>(mut get_neighbor: F) -> Self
    where
        F: FnMut(NeighborFace) -> Option<&'a Fld>,
        Fld: Field<T> + 'a,
    {
        let mut result = Self::default();
        for face in NeighborFace::ALL {
            if let Some(field) = get_neighbor(face) {
                result.neighbors[face as usize] = Some(NeighborSlice::from_field(field, face));
            }
        }
        result
    }

    /// Gather neighbor data using a closure that returns an optional slice for each face.
    pub fn gather_slices<F>(mut get_slice: F) -> Self
    where
        F: FnMut(NeighborFace) -> Option<NeighborSlice<T>>,
    {
        let mut result = Self::default();
        for face in NeighborFace::ALL {
            result.neighbors[face as usize] = get_slice(face);
        }
        result
    }

    #[inline]
    pub fn sample(&self, voxel: IVec3, field_size: IVec3) -> Option<T> {
        // Classify how many axes are out of bounds
        let oob_x = if voxel.x < 0 {
            -1
        } else if voxel.x >= field_size.x {
            1
        } else {
            0
        };
        let oob_y = if voxel.y < 0 {
            -1
        } else if voxel.y >= field_size.y {
            1
        } else {
            0
        };
        let oob_z = if voxel.z < 0 {
            -1
        } else if voxel.z >= field_size.z {
            1
        } else {
            0
        };

        let oob_count = (oob_x != 0) as u8 + (oob_y != 0) as u8 + (oob_z != 0) as u8;

        match oob_count {
            0 => None,
            1 => {
                let face = match (oob_x, oob_y, oob_z) {
                    (-1, 0, 0) => NeighborFace::NegX,
                    (1, 0, 0) => NeighborFace::PosX,
                    (0, -1, 0) => NeighborFace::NegY,
                    (0, 1, 0) => NeighborFace::PosY,
                    (0, 0, -1) => NeighborFace::NegZ,
                    (0, 0, 1) => NeighborFace::PosZ,
                    _ => return None,
                };
                let (a, b, depth) = face.voxel_to_slice_coords(voxel, field_size)?;
                self.neighbors[face as usize]
                    .as_ref()
                    .and_then(|slice| slice.get(a, b, depth))
            }
            2 => {
                let edge = match (oob_x, oob_y, oob_z) {
                    (-1, -1, 0) => NeighborEdge::NegXNegY,
                    (-1, 1, 0) => NeighborEdge::NegXPosY,
                    (1, -1, 0) => NeighborEdge::PosXNegY,
                    (1, 1, 0) => NeighborEdge::PosXPosY,
                    (-1, 0, -1) => NeighborEdge::NegXNegZ,
                    (-1, 0, 1) => NeighborEdge::NegXPosZ,
                    (1, 0, -1) => NeighborEdge::PosXNegZ,
                    (1, 0, 1) => NeighborEdge::PosXPosZ,
                    (0, -1, -1) => NeighborEdge::NegYNegZ,
                    (0, -1, 1) => NeighborEdge::NegYPosZ,
                    (0, 1, -1) => NeighborEdge::PosYNegZ,
                    (0, 1, 1) => NeighborEdge::PosYPosZ,
                    _ => return None,
                };
                let (a, du, dv) = edge.voxel_to_edge_coords(voxel, field_size)?;
                self.edges[edge as usize]
                    .as_ref()
                    .and_then(|slice| slice.get(a, du, dv))
            }
            3 => {
                let corner = match (oob_x, oob_y, oob_z) {
                    (-1, -1, -1) => NeighborCorner::NegXNegYNegZ,
                    (1, -1, -1) => NeighborCorner::PosXNegYNegZ,
                    (-1, 1, -1) => NeighborCorner::NegXPosYNegZ,
                    (1, 1, -1) => NeighborCorner::PosXPosYNegZ,
                    (-1, -1, 1) => NeighborCorner::NegXNegYPosZ,
                    (1, -1, 1) => NeighborCorner::PosXNegYPosZ,
                    (-1, 1, 1) => NeighborCorner::NegXPosYPosZ,
                    (1, 1, 1) => NeighborCorner::PosXPosYPosZ,
                    _ => return None,
                };
                let (dx, dy, dz) = corner.voxel_to_corner_coords(voxel, field_size)?;
                self.corners[corner as usize]
                    .as_ref()
                    .and_then(|slice| slice.get(dx, dy, dz))
            }
            _ => None,
        }
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

    /// Count how many face neighbors have data.
    pub fn neighbor_count(&self) -> usize {
        self.neighbors.iter().filter(|n| n.is_some()).count()
    }

    /// Count total neighbor slots with data (faces + edges + corners).
    pub fn total_neighbor_count(&self) -> usize {
        self.neighbors.iter().filter(|n| n.is_some()).count()
            + self.edges.iter().filter(|n| n.is_some()).count()
            + self.corners.iter().filter(|n| n.is_some()).count()
    }
}

/// Backward compatibility alias for f32 neighbor fields.
#[deprecated(since = "0.19.0", note = "Renamed to NeighborFields<f32>")]
pub type NeighborIsoFields = NeighborFields<f32>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbor_fields_sample() {
        let mut fields: NeighborFields<f32> = NeighborFields::default();

        fields.neighbors[NeighborFace::PosX as usize] = Some(NeighborSlice::from_sampler(
            NeighborFace::PosX,
            uvec3(32, 32, 32),
            NEIGHBOR_DEPTH,
            |_, _, _| -0.5,
        ));

        assert_eq!(
            fields.sample(ivec3(32, 5, 5), ivec3(32, 32, 32)),
            Some(-0.5)
        );
        // In-bounds → None (caller should use field directly)
        assert_eq!(fields.sample(ivec3(5, 5, 5), ivec3(32, 32, 32)), None);
        // OOB beyond stored depth → None (not 0.0!)
        assert_eq!(fields.sample(ivec3(35, 5, 5), ivec3(32, 32, 32)), None);
    }

    #[test]
    fn test_lod_aware_depth() {
        let depth = neighbor_depth_for_step(2);
        assert_eq!(depth, 3);

        let size = ivec3(32, 32, 32);
        let mut fields: NeighborFields<f32> = NeighborFields::default();

        fields.neighbors[NeighborFace::PosX as usize] = Some(NeighborSlice::from_sampler(
            NeighborFace::PosX,
            uvec3(32, 32, 32),
            depth,
            |_, _, _| -0.5,
        ));

        // All depths 0..2 covered by depth=3
        assert_eq!(fields.sample(ivec3(32, 5, 5), size), Some(-0.5)); // depth 0
        assert_eq!(fields.sample(ivec3(33, 5, 5), size), Some(-0.5)); // depth 1
        assert_eq!(fields.sample(ivec3(34, 5, 5), size), Some(-0.5)); // depth 2
        // depth 3 = beyond stored range → None
        assert_eq!(fields.sample(ivec3(35, 5, 5), size), None);
    }

    #[test]
    fn test_sample_dispatches_to_edge() {
        let size = ivec3(32, 32, 32);
        let depth = neighbor_depth_for_step(2);
        let mut fields: NeighborFields<f32> = NeighborFields::default();

        fields.edges[NeighborEdge::PosXPosY as usize] = Some(NeighborEdgeSlice {
            data: vec![-0.75; (32 * depth * depth) as usize],
            axis_len: 32,
            depth,
        });

        assert_eq!(fields.sample(ivec3(32, 32, 10), size), Some(-0.75));
        assert_eq!(fields.sample(ivec3(34, 34, 10), size), Some(-0.75));
        // No face data → None
        assert_eq!(fields.sample(ivec3(32, 5, 5), size), None);
    }

    #[test]
    fn test_sample_dispatches_to_corner() {
        let size = ivec3(32, 32, 32);
        let depth = neighbor_depth_for_step(2);
        let mut fields: NeighborFields<f32> = NeighborFields::default();

        fields.corners[NeighborCorner::PosXPosYPosZ as usize] = Some(NeighborCornerSlice {
            data: vec![-0.25; (depth * depth * depth) as usize],
            depth,
        });

        assert_eq!(fields.sample(ivec3(32, 32, 32), size), Some(-0.25));
        assert_eq!(fields.sample(ivec3(34, 34, 34), size), Some(-0.25));
        // Only 2 axes OOB → edge (no edge data)
        assert_eq!(fields.sample(ivec3(32, 32, 10), size), None);
    }
}
