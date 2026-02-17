//! Surface Nets mesh generation from sculptable fields.
//!
//! This module implements the Surface Nets algorithm for generating smooth meshes
//! from any field implementing [`Sculptable`].

use std::marker::PhantomData;

use crate::field_csg::IsoConvertible;
use crate::{NULL_VERTEX, neighbor::NeighborFields, sculptable::Sculptable};
use bevy::{
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

/// World-space size of meshes generated from sculptable fields.
///
/// This resource controls the scale of generated meshes. A field always has
/// [`FIELD_SIZE`](crate::FIELD_SIZE) voxels, but this determines how large
/// that maps to in world coordinates.
///
/// # Default
/// 10×10×10 world units
///
/// # Example
///
/// ```
/// use bevy::prelude::*;
/// use bevy_sculpter::prelude::*;
///
/// // Make each chunk mesh 20 world units on each side
/// App::new().insert_resource(MeshSize(vec3(20., 20., 20.)));
/// ```
#[derive(Resource, Clone, Copy, Deref, DerefMut, Debug)]
pub struct MeshSize(pub Vec3);

impl Default for MeshSize {
    fn default() -> Self {
        Self(vec3(10., 10., 10.))
    }
}

/// Backward compatibility alias.
#[deprecated(since = "0.2.0", note = "Renamed to MeshSize")]
pub type DensityFieldMeshSize = MeshSize;

/// Sampler that reads values from a Sculptable field and neighbors,
/// converting to iso via `Sculptable::to_iso`.

struct IsoSampler<'a, F, T>
where
    F: Sculptable<T>,
    T: IsoConvertible + Send + Sync + 'static,
{
    field: &'a F,
    neighbors: &'a NeighborFields<T>,
    _marker: PhantomData<T>,
}

impl<'a, F, T> IsoSampler<'a, F, T>
where
    F: Sculptable<T>,
    T: IsoConvertible + Send + Sync + 'static,
{
    fn new(field: &'a F, neighbors: &'a NeighborFields<T>) -> Self {
        Self {
            field,
            neighbors,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn sample(&self, x: i32, y: i32, z: i32) -> f32 {
        // 1. In bounds — read from local field
        if let Some(value) = self.field.get_signed(x, y, z) {
            return value.to_iso();
        }

        // 2. Out of bounds — read from neighbor (face, edge, or corner)
        if let Some(value) = self.neighbors.sample(ivec3(x, y, z), F::SIZE.as_ivec3()) {
            return value.to_iso();
        }

        // 3. No neighbor data available — return default (outside/air)
        F::DEFAULT.to_iso()
    }

    #[inline]
    fn sample_ivec3(&self, pos: IVec3) -> f32 {
        self.sample(pos.x, pos.y, pos.z)
    }
}

// And update generate_mesh_cpu's bound:
pub fn generate_mesh_cpu<F, T>(
    field: &F,
    neighbors: &NeighborFields<T>,
    mesh_size: Vec3,
) -> Option<Mesh>
where
    F: Sculptable<T>,
    T: IsoConvertible + Send + Sync + 'static,
{
    let sampler = IsoSampler::new(field, neighbors);
    let field_size = F::SIZE;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Extended size for boundary vertices
    let ext_size = field_size + UVec3::ONE;
    let ext_vol = (ext_size.x * ext_size.y * ext_size.z) as usize;
    let mut vtx_lookup = vec![NULL_VERTEX; ext_vol];

    let ext_idx = |x: u32, y: u32, z: u32| -> usize {
        (x + y * ext_size.x + z * ext_size.x * ext_size.y) as usize
    };

    const CORNERS: [IVec3; 8] = [
        IVec3::new(0, 0, 0),
        IVec3::new(1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(1, 1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(1, 0, 1),
        IVec3::new(0, 1, 1),
        IVec3::new(1, 1, 1),
    ];

    const CORNER_VECS: [Vec3; 8] = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];

    const EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ];

    let scale = mesh_size / field_size.as_vec3();
    let grid_to_world = |p: Vec3| -> [f32; 3] {
        let w = p * scale;
        [w.x, w.y, w.z]
    };

    // Pass 1: Generate vertices
    for z in 0..=field_size.z {
        for y in 0..=field_size.y {
            for x in 0..=field_size.x {
                let voxel = ivec3(x as i32, y as i32, z as i32);

                let mut corner_iso = [0.0f32; 8];
                let mut num_neg = 0;

                for (i, off) in CORNERS.iter().enumerate() {
                    corner_iso[i] = sampler.sample_ivec3(voxel + *off);
                    if corner_iso[i] < 0.0 {
                        num_neg += 1;
                    }
                }

                let stride = ext_idx(x, y, z);

                if num_neg == 0 || num_neg == 8 {
                    vtx_lookup[stride] = NULL_VERTEX;
                    continue;
                }

                // Centroid of edge crossings
                let mut sum = Vec3::ZERO;
                let mut count = 0.0;

                for edge in &EDGES {
                    let d1 = corner_iso[edge[0]];
                    let d2 = corner_iso[edge[1]];

                    if (d1 < 0.0) != (d2 < 0.0) {
                        let t = d1 / (d1 - d2);
                        let c1 = CORNER_VECS[edge[0]];
                        let c2 = CORNER_VECS[edge[1]];
                        sum += c1 + t * (c2 - c1);
                        count += 1.0;
                    }
                }

                let centroid = if count > 0.0 {
                    sum / count
                } else {
                    Vec3::splat(0.5)
                };
                let grid_pos = voxel.as_vec3() + centroid;
                let world_pos = grid_to_world(grid_pos);

                // Normal via central differences
                let dx = sampler.sample(voxel.x + 1, voxel.y, voxel.z)
                    - sampler.sample(voxel.x - 1, voxel.y, voxel.z);
                let dy = sampler.sample(voxel.x, voxel.y + 1, voxel.z)
                    - sampler.sample(voxel.x, voxel.y - 1, voxel.z);
                let dz = sampler.sample(voxel.x, voxel.y, voxel.z + 1)
                    - sampler.sample(voxel.x, voxel.y, voxel.z - 1);

                let grad = vec3(dx, dy, dz);
                let normal = if grad.length_squared() > 0.0001 {
                    grad.normalize()
                } else {
                    Vec3::Y
                };

                vtx_lookup[stride] = positions.len() as u32;
                positions.push(world_pos);
                normals.push(normal.into());
            }
        }
    }

    if positions.is_empty() {
        return None;
    }

    // Pass 2: Generate quads
    let ext_sx = 1usize;
    let ext_sy = ext_size.x as usize;
    let ext_sz = (ext_size.x * ext_size.y) as usize;

    for z in 1..=field_size.z {
        for y in 1..=field_size.y {
            for x in 1..=field_size.x {
                let stride = ext_idx(x, y, z);
                let v0 = vtx_lookup[stride];

                if v0 == NULL_VERTEX {
                    continue;
                }

                let voxel = ivec3(x as i32, y as i32, z as i32);

                let d0 = sampler.sample_ivec3(voxel);
                let dx = sampler.sample_ivec3(voxel + IVec3::X);
                let dy = sampler.sample_ivec3(voxel + IVec3::Y);
                let dz = sampler.sample_ivec3(voxel + IVec3::Z);

                // X-axis edge
                if (d0 < 0.0) != (dx < 0.0) {
                    let v1 = vtx_lookup[stride - ext_sy];
                    let v2 = vtx_lookup[stride - ext_sz];
                    let v3 = vtx_lookup[stride - ext_sy - ext_sz];

                    if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
                        if d0 < 0.0 {
                            indices.extend_from_slice(&[v0, v1, v3, v0, v3, v2]);
                        } else {
                            indices.extend_from_slice(&[v0, v3, v1, v0, v2, v3]);
                        }
                    }
                }

                // Y-axis edge
                if (d0 < 0.0) != (dy < 0.0) {
                    let v1 = vtx_lookup[stride - ext_sx];
                    let v2 = vtx_lookup[stride - ext_sz];
                    let v3 = vtx_lookup[stride - ext_sx - ext_sz];

                    if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
                        if d0 < 0.0 {
                            indices.extend_from_slice(&[v0, v2, v3, v0, v3, v1]);
                        } else {
                            indices.extend_from_slice(&[v0, v3, v2, v0, v1, v3]);
                        }
                    }
                }

                // Z-axis edge
                if (d0 < 0.0) != (dz < 0.0) {
                    let v1 = vtx_lookup[stride - ext_sx];
                    let v2 = vtx_lookup[stride - ext_sy];
                    let v3 = vtx_lookup[stride - ext_sx - ext_sy];

                    if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
                        if d0 < 0.0 {
                            indices.extend_from_slice(&[v0, v1, v3, v0, v3, v2]);
                        } else {
                            indices.extend_from_slice(&[v0, v3, v1, v0, v2, v3]);
                        }
                    }
                }
            }
        }
    }

    if indices.is_empty() {
        return None;
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::RENDER_WORLD | bevy::asset::RenderAssetUsages::MAIN_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));

    Some(mesh)
}
