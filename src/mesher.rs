//! Surface Nets mesh generation from density fields.
//!
//! This module implements the Surface Nets algorithm for generating smooth meshes
//! from signed distance fields. Unlike Marching Cubes, Surface Nets produces
//! vertex positions that lie on the actual isosurface, resulting in smoother meshes.
//!
//! # Algorithm Overview
//!
//! 1. For each voxel that contains a surface crossing (some corners inside, some outside):
//!    - Find all edges that cross the isosurface
//!    - Compute the centroid of the crossing points
//!    - Place a vertex at that centroid
//!
//! 2. For each edge that crosses the isosurface:
//!    - Connect the four adjacent voxel vertices into a quad
//!
//! # Seamless Chunk Boundaries
//!
//! The implementation extends the vertex grid by one in each positive direction,
//! using [`NeighborDensityFields`] data to sample beyond chunk boundaries.
//! This allows quads to be generated that span chunk boundaries.

use bevy::{
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use crate::{density_field::DefaultIsoField, field::Field, neighbor::NeighborDensityFields};
pub const NULL_VERTEX: u32 = u32::MAX;

/// World-space size of the mesh generated from a density field.
///
/// This resource controls the scale of generated meshes. A density field
/// always has [`DensityField::SIZE`] voxels, but this determines how large
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
/// // Make each chunk 20 world units on each side
/// App::new()
///     .insert_resource(DensityFieldMeshSize(vec3(20., 20., 20.)));
/// ```
#[derive(Resource, Clone, Copy, Deref, DerefMut, Debug)]
pub struct DensityFieldMeshSize(pub Vec3);

impl Default for DensityFieldMeshSize {
    fn default() -> Self {
        Self(vec3(10., 10., 10.))
    }
}

/// Sampler that reads from a field and its neighbors seamlessly.
///
/// This struct encapsulates the logic for sampling density values both
/// within the local field and from neighboring chunks.
struct FieldSampler<'a> {
    field: &'a DefaultIsoField,
    neighbors: &'a NeighborDensityFields,
}

impl<'a> FieldSampler<'a> {
    fn new(field: &'a DefaultIsoField, neighbors: &'a NeighborDensityFields) -> Self {
        Self { field, neighbors }
    }

    /// Sample density at signed coordinates, checking neighbors if out of bounds.
    #[inline]
    fn sample(&self, x: i32, y: i32, z: i32) -> f32 {
        // Try local field first
        if let Some(value) = self.field.get_signed(x, y, z) {
            return value;
        }

        // Try neighbors
        if let Some(value) = self.neighbors.sample_for::<DefaultIsoField>(ivec3(x, y, z)) {
            return value;
        }

        // Fallback: clamp to nearest in-bounds voxel
        let size = DefaultIsoField::SIZE.as_ivec3();
        let clamped_x = x.clamp(0, size.x - 1) as u32;
        let clamped_y = y.clamp(0, size.y - 1) as u32;
        let clamped_z = z.clamp(0, size.z - 1) as u32;
        self.field.get(clamped_x, clamped_y, clamped_z)
    }

    /// Sample at IVec3 position.
    #[inline]
    fn sample_ivec3(&self, pos: IVec3) -> f32 {
        self.sample(pos.x, pos.y, pos.z)
    }
}

/// Generates a mesh from a density field using the Surface Nets algorithm.
///
/// This function runs entirely on the CPU and produces a Bevy [`Mesh`] with
/// position and normal attributes.
///
/// # Arguments
/// * `field` - The density field to mesh
/// * `neighbors` - Cached neighbor data for seamless boundaries (can be empty)
/// * `mesh_size` - World-space dimensions of the output mesh
///
/// # Returns
/// `Some(Mesh)` if any surface was found, `None` if the field is entirely inside or outside.
///
/// # Example
///
/// ```ignore
/// let mesh = generate_mesh_cpu(&field, &neighbors, vec3(10., 10., 10.));
/// if let Some(mesh) = mesh {
///     commands.spawn(Mesh3d(meshes.add(mesh)));
/// }
/// ```
pub fn generate_mesh_cpu(
    field: &DefaultIsoField,
    neighbors: &NeighborDensityFields,
    mesh_size: Vec3,
) -> Option<Mesh> {
    let sampler = FieldSampler::new(field, neighbors);
    let field_size = DefaultIsoField::SIZE;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Extended size: we generate vertices for voxels [0, SIZE] inclusive
    // The extra layer at SIZE uses neighbor data and allows seamless stitching
    let extended_size = field_size + UVec3::ONE;
    let extended_volume = (extended_size.x * extended_size.y * extended_size.z) as usize;
    let mut vertex_lookup = vec![NULL_VERTEX; extended_volume];

    // Index into extended lookup table
    let ext_index = |x: u32, y: u32, z: u32| -> usize {
        (x + y * extended_size.x + z * extended_size.x * extended_size.y) as usize
    };

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
        [0, 1],
        [0, 2],
        [0, 4], // edges from corner 0
        [1, 3],
        [1, 5], // edges from corner 1
        [2, 3],
        [2, 6], // edges from corner 2
        [3, 7], // edge from corner 3
        [4, 5],
        [4, 6], // edges from corner 4
        [5, 7], // edge from corner 5
        [6, 7], // edge from corner 6
    ];

    let scale = mesh_size / field_size.as_vec3();
    let grid_to_world = |grid_pos: Vec3| -> [f32; 3] {
        let world = grid_pos * scale;
        [world.x, world.y, world.z]
    };

    // Pass 1: Generate vertices for each voxel that contains a surface
    for z in 0..=field_size.z {
        for y in 0..=field_size.y {
            for x in 0..=field_size.x {
                let voxel = ivec3(x as i32, y as i32, z as i32);

                // Sample the 8 corners of this voxel's cube
                let mut corner_dists = [0.0f32; 8];
                let mut num_negative = 0;

                for (i, offset) in CORNERS.iter().enumerate() {
                    corner_dists[i] = sampler.sample_ivec3(voxel + *offset);
                    if corner_dists[i] < 0.0 {
                        num_negative += 1;
                    }
                }

                let stride = ext_index(x, y, z);

                // If all corners are inside or all outside, no surface here
                if num_negative == 0 || num_negative == 8 {
                    vertex_lookup[stride] = NULL_VERTEX;
                    continue;
                }

                // Calculate centroid of edge intersection points (Surface Nets method)
                let mut sum = Vec3::ZERO;
                let mut count = 0.0;

                for edge in &EDGES {
                    let d1 = corner_dists[edge[0]];
                    let d2 = corner_dists[edge[1]];

                    // Edge crosses the surface if signs differ
                    if (d1 < 0.0) != (d2 < 0.0) {
                        // Linear interpolation to find crossing point
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

                // Convert to world position
                let grid_pos = voxel.as_vec3() + centroid;
                let world_pos = grid_to_world(grid_pos);

                // Calculate normal via central differences gradient
                let dx = sampler.sample(voxel.x + 1, voxel.y, voxel.z)
                    - sampler.sample(voxel.x - 1, voxel.y, voxel.z);
                let dy = sampler.sample(voxel.x, voxel.y + 1, voxel.z)
                    - sampler.sample(voxel.x, voxel.y - 1, voxel.z);
                let dz = sampler.sample(voxel.x, voxel.y, voxel.z + 1)
                    - sampler.sample(voxel.x, voxel.y, voxel.z - 1);

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

    // Pass 2: Generate quads between adjacent voxels that share an edge crossing
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

                let d0 = sampler.sample_ivec3(voxel);
                let dx = sampler.sample_ivec3(voxel + IVec3::X);
                let dy = sampler.sample_ivec3(voxel + IVec3::Y);
                let dz = sampler.sample_ivec3(voxel + IVec3::Z);

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
