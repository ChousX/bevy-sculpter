use bevy::{
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use crate::{
    DENSITY_FIELD_SIZE, FIELD_VOLUME, NULL_VERTEX,
    density_field::DensityField,
    neighbor::{NeighborDensityFields, NeighborFace},
};

/// World-space size of mesh generated from density field
#[derive(Resource, Clone, Copy, Deref, DerefMut, Debug)]
pub struct DensityFieldMeshSize(pub Vec3);

impl Default for DensityFieldMeshSize {
    fn default() -> Self {
        Self(vec3(10., 10., 10.))
    }
}

/// Generate mesh on CPU using Surface Nets algorithm
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

    // Sample function that handles neighbor lookups
    let sample = |x: i32, y: i32, z: i32| -> f32 {
        // In bounds - direct sample
        if DensityField::in_bounds(x, y, z) {
            return field.get(x as u32, y as u32, z as u32);
        }

        // Out of bounds - check neighbors
        // -X neighbor (x < 0)
        if x < 0
            && y >= 0
            && z >= 0
            && (y as u32) < DENSITY_FIELD_SIZE.y
            && (z as u32) < DENSITY_FIELD_SIZE.z
        {
            if let Some(ref slice) = neighbors.neighbors[NeighborFace::NegX as usize] {
                return slice.get(y as u32, z as u32);
            }
        }

        // +X neighbor (x >= SIZE)
        if x >= DENSITY_FIELD_SIZE.x as i32
            && y >= 0
            && z >= 0
            && (y as u32) < DENSITY_FIELD_SIZE.y
            && (z as u32) < DENSITY_FIELD_SIZE.z
        {
            if let Some(ref slice) = neighbors.neighbors[NeighborFace::PosX as usize] {
                return slice.get(y as u32, z as u32);
            }
        }

        // -Y neighbor (y < 0)
        if y < 0
            && x >= 0
            && z >= 0
            && (x as u32) < DENSITY_FIELD_SIZE.x
            && (z as u32) < DENSITY_FIELD_SIZE.z
        {
            if let Some(ref slice) = neighbors.neighbors[NeighborFace::NegY as usize] {
                return slice.get(x as u32, z as u32);
            }
        }

        // +Y neighbor (y >= SIZE)
        if y >= DENSITY_FIELD_SIZE.y as i32
            && x >= 0
            && z >= 0
            && (x as u32) < DENSITY_FIELD_SIZE.x
            && (z as u32) < DENSITY_FIELD_SIZE.z
        {
            if let Some(ref slice) = neighbors.neighbors[NeighborFace::PosY as usize] {
                return slice.get(x as u32, z as u32);
            }
        }

        // -Z neighbor (z < 0)
        if z < 0
            && x >= 0
            && y >= 0
            && (x as u32) < DENSITY_FIELD_SIZE.x
            && (y as u32) < DENSITY_FIELD_SIZE.y
        {
            if let Some(ref slice) = neighbors.neighbors[NeighborFace::NegZ as usize] {
                return slice.get(x as u32, y as u32);
            }
        }

        // +Z neighbor (z >= SIZE)
        if z >= DENSITY_FIELD_SIZE.z as i32
            && x >= 0
            && y >= 0
            && (x as u32) < DENSITY_FIELD_SIZE.x
            && (y as u32) < DENSITY_FIELD_SIZE.y
        {
            if let Some(ref slice) = neighbors.neighbors[NeighborFace::PosZ as usize] {
                return slice.get(x as u32, y as u32);
            }
        }

        1.0 // Outside (no neighbor data available)
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
    // We iterate over [0, SIZE] inclusive - the extra layer at SIZE uses neighbor data
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
    //
    // Surface Nets generates one quad per edge that crosses the isosurface.
    // Each edge is "owned" by one of the four voxels it connects.
    // We use the convention that the voxel with the MINIMUM coordinates owns the edge.
    //
    // With extended vertices [0, SIZE], we can now generate quads at the high boundary
    // that connect this chunk's mesh to the neighbor's mesh seamlessly.
    //
    // We iterate over [1, SIZE] for quad generation:
    // - Skip 0 because those edges are owned by the -X/-Y/-Z neighbor chunks
    // - Include SIZE because we have vertices there now (using neighbor data)

    let ext_stride_x = 1usize;
    let ext_stride_y = extended_size.x as usize;
    let ext_stride_z = (extended_size.x * extended_size.y) as usize;

    // Iterate over [1, SIZE] inclusive - we need vertices at (x-1), (y-1), (z-1)
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

                // Get SDF values at current position and +1 in each axis
                let d0 = sample(ix, iy, iz);
                let dx = sample(ix + 1, iy, iz);
                let dy = sample(ix, iy + 1, iz);
                let dz = sample(ix, iy, iz + 1);

                // X-axis edge (between x and x+1): need voxels (x,y,z), (x,y-1,z), (x,y,z-1), (x,y-1,z-1)
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

                // Y-axis edge (between y and y+1): need voxels (x,y,z), (x-1,y,z), (x,y,z-1), (x-1,y,z-1)
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

                // Z-axis edge (between z and z+1): need voxels (x,y,z), (x-1,y,z), (x,y-1,z), (x-1,y-1,z)
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
