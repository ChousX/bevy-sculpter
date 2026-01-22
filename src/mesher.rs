//! Surface Nets mesh generation from density fields.
//!
//! Generic over any field type implementing [`Sculptable`].

use bevy::{
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use crate::{
    neighbor::NeighborFields,
    sculptable::Sculptable,
};

pub const NULL_VERTEX: u32 = u32::MAX;

/// World-space size of the mesh generated from a density field.
#[derive(Resource, Clone, Copy, Deref, DerefMut, Debug)]
pub struct DensityFieldMeshSize(pub Vec3);

impl Default for DensityFieldMeshSize {
    fn default() -> Self {
        Self(vec3(10., 10., 10.))
    }
}

/// Sampler that reads from a sculptable field and its neighbors seamlessly.
struct FieldSampler<'a, T: Copy + Clone + Default + Send + Sync + 'static, F: Sculptable<T>> {
    field: &'a F,
    neighbors: &'a NeighborFields<T>,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: Copy + Clone + Default + Send + Sync + 'static, F: Sculptable<T>> FieldSampler<'a, T, F> {
    fn new(field: &'a F, neighbors: &'a NeighborFields<T>) -> Self {
        Self {
            field,
            neighbors,
            _marker: std::marker::PhantomData,
        }
    }

    /// Sample iso value at signed coordinates, checking neighbors if out of bounds.
    #[inline]
    fn sample_iso(&self, x: i32, y: i32, z: i32) -> f32 {
        // Try local field first
        if let Some(iso) = self.field.sample_iso_ivec3(ivec3(x, y, z)) {
            return iso;
        }

        // Try neighbors - get raw T and convert to iso
        if let Some(raw) = self.neighbors.sample_for::<F>(ivec3(x, y, z)) {
            return F::to_iso(raw);
        }

        // Fallback: clamp to nearest in-bounds voxel
        let size = F::SIZE.as_ivec3();
        let clamped = ivec3(
            x.clamp(0, size.x - 1),
            y.clamp(0, size.y - 1),
            z.clamp(0, size.z - 1),
        );
        self.field.sample_iso(clamped.x as u32, clamped.y as u32, clamped.z as u32)
    }

    #[inline]
    fn sample_iso_ivec3(&self, pos: IVec3) -> f32 {
        self.sample_iso(pos.x, pos.y, pos.z)
    }
}

// Cube corners (offsets from voxel origin)
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

// 12 edges of a cube, each connecting two corners
const EDGES: [[usize; 2]; 12] = [
    [0, 1], [0, 2], [0, 4],
    [1, 3], [1, 5],
    [2, 3], [2, 6],
    [3, 7],
    [4, 5], [4, 6],
    [5, 7],
    [6, 7],
];

/// Generates a mesh from any sculptable field using Surface Nets.
///
/// # Type Parameters
/// * `T` - The storage type of the field
/// * `F` - The field type implementing `Sculptable<T>`
///
/// # Arguments
/// * `field` - The sculptable field to mesh
/// * `neighbors` - Cached neighbor data for seamless boundaries
/// * `mesh_size` - World-space dimensions of the output mesh
pub fn generate_mesh<T, F>(
    field: &F,
    neighbors: &NeighborFields<T>,
    mesh_size: Vec3,
) -> Option<Mesh>
where
    T: Copy + Clone + Default + Send + Sync + 'static,
    F: Sculptable<T>,
{
    let sampler = FieldSampler::new(field, neighbors);
    let field_size = F::SIZE;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let extended_size = field_size + UVec3::ONE;
    let extended_volume = (extended_size.x * extended_size.y * extended_size.z) as usize;
    let mut vertex_lookup = vec![NULL_VERTEX; extended_volume];

    let ext_index = |x: u32, y: u32, z: u32| -> usize {
        (x + y * extended_size.x + z * extended_size.x * extended_size.y) as usize
    };

    let scale = mesh_size / field_size.as_vec3();
    let grid_to_world = |grid_pos: Vec3| -> [f32; 3] {
        let world = grid_pos * scale;
        [world.x, world.y, world.z]
    };

    // Pass 1: Generate vertices
    for z in 0..=field_size.z {
        for y in 0..=field_size.y {
            for x in 0..=field_size.x {
                let voxel = ivec3(x as i32, y as i32, z as i32);

                let mut corner_dists = [0.0f32; 8];
                let mut num_negative = 0;

                for (i, offset) in CORNERS.iter().enumerate() {
                    corner_dists[i] = sampler.sample_iso_ivec3(voxel + *offset);
                    if corner_dists[i] < 0.0 {
                        num_negative += 1;
                    }
                }

                let stride = ext_index(x, y, z);

                if num_negative == 0 || num_negative == 8 {
                    vertex_lookup[stride] = NULL_VERTEX;
                    continue;
                }

                // Surface Nets: centroid of edge crossing points
                let mut sum = Vec3::ZERO;
                let mut count = 0.0;

                for edge in &EDGES {
                    let d1 = corner_dists[edge[0]];
                    let d2 = corner_dists[edge[1]];

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
                let dx = sampler.sample_iso(voxel.x + 1, voxel.y, voxel.z)
                    - sampler.sample_iso(voxel.x - 1, voxel.y, voxel.z);
                let dy = sampler.sample_iso(voxel.x, voxel.y + 1, voxel.z)
                    - sampler.sample_iso(voxel.x, voxel.y - 1, voxel.z);
                let dz = sampler.sample_iso(voxel.x, voxel.y, voxel.z + 1)
                    - sampler.sample_iso(voxel.x, voxel.y, voxel.z - 1);

                let gradient = vec3(dx, dy, dz);
                let normal = if gradient.length_squared() > 0.0001 {
                    gradient.normalize()
                } else {
                    Vec3::Y
                };

                vertex_lookup[stride] = positions.len() as u32;
                positions.push(world_pos);
                normals.push(normal.into());
            }
        }
    }

    if positions.is_empty() {
        return None;
    }

    // Pass 2: Generate quads
    let ext_stride_x = 1usize;
    let ext_stride_y = extended_size.x as usize;
    let ext_stride_z = (extended_size.x * extended_size.y) as usize;

    for z in 1..=field_size.z {
        for y in 1..=field_size.y {
            for x in 1..=field_size.x {
                let stride = ext_index(x, y, z);
                let v0 = vertex_lookup[stride];

                if v0 == NULL_VERTEX {
                    continue;
                }

                let voxel = ivec3(x as i32, y as i32, z as i32);

                let d0 = sampler.sample_iso_ivec3(voxel);
                let dx = sampler.sample_iso_ivec3(voxel + IVec3::X);
                let dy = sampler.sample_iso_ivec3(voxel + IVec3::Y);
                let dz = sampler.sample_iso_ivec3(voxel + IVec3::Z);

                // X-axis edge
                if (d0 < 0.0) != (dx < 0.0) {
                    let v1 = vertex_lookup[stride - ext_stride_y];
                    let v2 = vertex_lookup[stride - ext_stride_z];
                    let v3 = vertex_lookup[stride - ext_stride_y - ext_stride_z];

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
                    let v1 = vertex_lookup[stride - ext_stride_x];
                    let v2 = vertex_lookup[stride - ext_stride_z];
                    let v3 = vertex_lookup[stride - ext_stride_x - ext_stride_z];

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
                    let v1 = vertex_lookup[stride - ext_stride_x];
                    let v2 = vertex_lookup[stride - ext_stride_y];
                    let v3 = vertex_lookup[stride - ext_stride_x - ext_stride_y];

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

/// Convenience alias for f32 density fields (the common case).
pub fn generate_mesh_cpu<F: Sculptable<f32>>(
    field: &F,
    neighbors: &NeighborFields<f32>,
    mesh_size: Vec3,
) -> Option<Mesh> {
    generate_mesh(field, neighbors, mesh_size)
}
