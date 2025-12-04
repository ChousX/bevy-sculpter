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

use crate::{
    DENSITY_FIELD_SIZE, NULL_VERTEX,
    density_field::DensityField,
    neighbor::{NeighborDensityFields, NeighborFace},
};

/// World-space size of the mesh generated from a density field.
///
/// This resource controls the scale of generated meshes. A density field
/// always has [`DENSITY_FIELD_SIZE`] voxels, but this determines how large
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
    field: &DensityField,
    neighbors: &NeighborDensityFields,
    mesh_size: Vec3,
) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Extended size: we generate vertices for voxels [0, SIZE] inclusive
    // The extra layer at SIZE uses neighbor data and allows seamless stitching
    let extended_size = DENSITY_FIELD_SIZE + UVec3::ONE;
    let extended_volume = (extended_size.x * extended_size.y * extended_size.z) as usize;
    let mut vertex_lookup = vec![NULL_VERTEX; extended_volume];

    // Index into extended lookup table
    let ext_index = |x: u32, y: u32, z: u32| -> usize {
        (x + y * extended_size.x + z * extended_size.x * extended_size.y) as usize
    };

    let size_x = DENSITY_FIELD_SIZE.x as i32;
    let size_y = DENSITY_FIELD_SIZE.y as i32;
    let size_z = DENSITY_FIELD_SIZE.z as i32;

    // Sample function that handles neighbor lookups with proper depth
    let sample = |x: i32, y: i32, z: i32| -> f32 {
        // In bounds - direct sample
        if x >= 0 && y >= 0 && z >= 0 && x < size_x && y < size_y && z < size_z {
            return field.get(x as u32, y as u32, z as u32);
        }

        // -X neighbor (x < 0)
        if x < 0
            && y >= 0
            && z >= 0
            && y < size_y
            && z < size_z
            && let Some(ref slice) = neighbors.neighbors[NeighborFace::NegX as usize]
        {
            let depth = (-1 - x) as u32;
            return slice.get(y as u32, z as u32, depth);
        }

        // +X neighbor (x >= SIZE)
        if x >= size_x
            && y >= 0
            && z >= 0
            && y < size_y
            && z < size_z
            && let Some(ref slice) = neighbors.neighbors[NeighborFace::PosX as usize]
        {
            let depth = (x - size_x) as u32;
            return slice.get(y as u32, z as u32, depth);
        }

        // -Y neighbor (y < 0)
        if y < 0
            && x >= 0
            && z >= 0
            && x < size_x
            && z < size_z
            && let Some(ref slice) = neighbors.neighbors[NeighborFace::NegY as usize]
        {
            let depth = (-1 - y) as u32;
            return slice.get(x as u32, z as u32, depth);
        }

        // +Y neighbor (y >= SIZE)
        if y >= size_y
            && x >= 0
            && z >= 0
            && x < size_x
            && z < size_z
            && let Some(ref slice) = neighbors.neighbors[NeighborFace::PosY as usize]
        {
            let depth = (y - size_y) as u32;
            return slice.get(x as u32, z as u32, depth);
        }

        // -Z neighbor (z < 0)
        if z < 0
            && x >= 0
            && y >= 0
            && x < size_x
            && y < size_y
            && let Some(ref slice) = neighbors.neighbors[NeighborFace::NegZ as usize]
        {
            let depth = (-1 - z) as u32;
            return slice.get(x as u32, y as u32, depth);
        }

        // +Z neighbor (z >= SIZE)
        if z >= size_z
            && x >= 0
            && y >= 0
            && x < size_x
            && y < size_y
            && let Some(ref slice) = neighbors.neighbors[NeighborFace::PosZ as usize]
        {
            let depth = (z - size_z) as u32;
            return slice.get(x as u32, y as u32, depth);
        }

        // Edge/corner cases involving multiple neighbors - return outside
        1.0
    };

    // Cube corners (offsets from voxel origin)
    const CORNERS: [[i32; 3]; 8] = [
        [0, 0, 0], // 0
        [1, 0, 0], // 1
        [0, 1, 0], // 2
        [1, 1, 0], // 3
        [0, 0, 1], // 4
        [1, 0, 1], // 5
        [0, 1, 1], // 6
        [1, 1, 1], // 7
    ];

    const CORNER_VECS: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
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

    let grid_to_world = |gx: f32, gy: f32, gz: f32| -> [f32; 3] {
        let scale = mesh_size / DENSITY_FIELD_SIZE.as_vec3();
        [gx * scale.x, gy * scale.y, gz * scale.z]
    };

    // Pass 1: Generate vertices for each voxel that contains a surface
    for z in 0..=DENSITY_FIELD_SIZE.z {
        for y in 0..=DENSITY_FIELD_SIZE.y {
            for x in 0..=DENSITY_FIELD_SIZE.x {
                let ix = x as i32;
                let iy = y as i32;
                let iz = z as i32;

                // Sample the 8 corners of this voxel's cube
                let mut corner_dists = [0.0f32; 8];
                let mut num_negative = 0;

                for (i, c) in CORNERS.iter().enumerate() {
                    corner_dists[i] = sample(ix + c[0], iy + c[1], iz + c[2]);
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
                let mut sum = [0.0f32; 3];
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
                        sum[0] += c1[0] + t * (c2[0] - c1[0]);
                        sum[1] += c1[1] + t * (c2[1] - c1[1]);
                        sum[2] += c1[2] + t * (c2[2] - c1[2]);
                        count += 1.0;
                    }
                }

                let centroid = if count > 0.0 {
                    [sum[0] / count, sum[1] / count, sum[2] / count]
                } else {
                    [0.5, 0.5, 0.5]
                };

                // Convert to world position
                let grid_pos = [
                    x as f32 + centroid[0],
                    y as f32 + centroid[1],
                    z as f32 + centroid[2],
                ];
                let world_pos = grid_to_world(grid_pos[0], grid_pos[1], grid_pos[2]);

                // Calculate normal via central differences gradient
                let dx = sample(ix + 1, iy, iz) - sample(ix - 1, iy, iz);
                let dy = sample(ix, iy + 1, iz) - sample(ix, iy - 1, iz);
                let dz = sample(ix, iy, iz + 1) - sample(ix, iy, iz - 1);
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                let normal = if len > 0.0001 {
                    [dx / len, dy / len, dz / len]
                } else {
                    [0.0, 1.0, 0.0]
                };

                vertex_lookup[stride] = positions.len() as u32;
                positions.push(world_pos);
                normals.push(normal);
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

    for z in 1..=DENSITY_FIELD_SIZE.z {
        for y in 1..=DENSITY_FIELD_SIZE.y {
            for x in 1..=DENSITY_FIELD_SIZE.x {
                let stride = ext_index(x, y, z);
                let v0 = vertex_lookup[stride];

                if v0 == NULL_VERTEX {
                    continue;
                }

                let ix = x as i32;
                let iy = y as i32;
                let iz = z as i32;

                let d0 = sample(ix, iy, iz);
                let dx = sample(ix + 1, iy, iz);
                let dy = sample(ix, iy + 1, iz);
                let dz = sample(ix, iy, iz + 1);

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
