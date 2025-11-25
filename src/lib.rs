// src/lib.rs - Surface Nets Plugin with GPU compute and mesh generation
use bevy::{
    mesh::Indices,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_graph::{RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
};
use chunky::{ChunkManager, ChunkPos};

pub mod prelude {
    pub use crate::{
        DENSITY_FIELD_SIZE, DensityField, DensityFieldDirty, DensityFieldMeshSize,
        NeighborDensityFields, SurfaceNetsPlugin,
    };
}

// ============================================================================
// Constants
// ============================================================================

/// Size of the density field grid per chunk (no padding)
pub const DENSITY_FIELD_SIZE: UVec3 = uvec3(32, 32, 32);

/// Total voxels per chunk
pub const FIELD_VOLUME: usize =
    (DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.y * DENSITY_FIELD_SIZE.z) as usize;

/// Max output sizes
pub const MAX_VERTICES: u32 = FIELD_VOLUME as u32;
pub const MAX_INDICES: u32 = MAX_VERTICES * 18;
/// Workgroup size
pub const WORKGROUP_SIZE: u32 = 4;

/// Null vertex marker
pub const NULL_VERTEX: u32 = u32::MAX;

// ============================================================================
// Plugin
// ============================================================================

pub struct SurfaceNetsPlugin;

impl Plugin for SurfaceNetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DensityFieldMeshSize>()
            .add_plugins(ExtractComponentPlugin::<DensityField>::default())
            .add_plugins(ExtractResourcePlugin::<DensityFieldMeshSize>::default())
            .add_systems(
                PostUpdate,
                (auto_mark_dirty, gather_neighbor_fields).chain(),
            )
            .add_systems(PreUpdate, (process_dirty_chunks, handle_mesh_readback));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SurfaceNetsPipeline>()
            .init_resource::<ExtractedFields>()
            .init_resource::<GpuBufferStore>()
            .add_systems(RenderStartup, setup_pipeline)
            .add_systems(
                Render,
                (extract_fields, prepare_buffers)
                    .chain()
                    .in_set(RenderSystems::Prepare),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(SurfaceNetsLabel, SurfaceNetsNode::default());
        render_graph.add_node_edge(SurfaceNetsLabel, bevy::render::graph::CameraDriverLabel);
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct SurfaceNetsLabel;

// ============================================================================
// Components & Resources
// ============================================================================

/// World-space size of mesh generated from density field
#[derive(Resource, Clone, Copy, Deref, DerefMut, Debug, ExtractResource)]
pub struct DensityFieldMeshSize(pub Vec3);

impl Default for DensityFieldMeshSize {
    fn default() -> Self {
        Self(vec3(10., 10., 10.))
    }
}

/// The density field (SDF). Negative = inside, Positive = outside.
#[derive(Component, ExtractComponent, Clone, Deref, DerefMut, Debug)]
pub struct DensityField(pub Vec<f32>);

impl Default for DensityField {
    fn default() -> Self {
        Self(vec![1.0; FIELD_VOLUME]) // All outside
    }
}

impl DensityField {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn filled(value: f32) -> Self {
        Self(vec![value; FIELD_VOLUME])
    }

    #[inline]
    pub fn index(x: u32, y: u32, z: u32) -> usize {
        (x + y * DENSITY_FIELD_SIZE.x + z * DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.y) as usize
    }

    #[inline]
    pub fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < DENSITY_FIELD_SIZE.x
            && (y as u32) < DENSITY_FIELD_SIZE.y
            && (z as u32) < DENSITY_FIELD_SIZE.z
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        if x < DENSITY_FIELD_SIZE.x && y < DENSITY_FIELD_SIZE.y && z < DENSITY_FIELD_SIZE.z {
            self.0[Self::index(x, y, z)] = value;
        }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        if x < DENSITY_FIELD_SIZE.x && y < DENSITY_FIELD_SIZE.y && z < DENSITY_FIELD_SIZE.z {
            self.0[Self::index(x, y, z)]
        } else {
            1.0 // Outside = exterior
        }
    }

    /// Get with signed coords (for neighbor sampling)
    #[inline]
    pub fn get_signed(&self, x: i32, y: i32, z: i32) -> Option<f32> {
        if Self::in_bounds(x, y, z) {
            Some(self.0[Self::index(x as u32, y as u32, z as u32)])
        } else {
            None
        }
    }

    pub fn fill_sphere(&mut self, center: Vec3, radius: f32) {
        for z in 0..DENSITY_FIELD_SIZE.z {
            for y in 0..DENSITY_FIELD_SIZE.y {
                for x in 0..DENSITY_FIELD_SIZE.x {
                    let pos = vec3(x as f32, y as f32, z as f32);
                    self.set(x, y, z, pos.distance(center) - radius);
                }
            }
        }
    }

    pub fn fill_centered_sphere(&mut self, radius: f32) {
        let center = DENSITY_FIELD_SIZE.as_vec3() / 2.0;
        self.fill_sphere(center, radius);
    }

    pub fn brush_sphere(&mut self, center: Vec3, radius: f32, add: bool) {
        let min = (center - Vec3::splat(radius + 1.0))
            .max(Vec3::ZERO)
            .as_ivec3();
        let max = (center + Vec3::splat(radius + 1.0))
            .min(DENSITY_FIELD_SIZE.as_vec3() - Vec3::ONE)
            .as_ivec3();

        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let pos = vec3(x as f32, y as f32, z as f32);
                    let sphere_sdf = pos.distance(center) - radius;
                    let current = self.get(x as u32, y as u32, z as u32);

                    let new_val = if add {
                        current.min(sphere_sdf)
                    } else {
                        current.max(-sphere_sdf)
                    };
                    self.set(x as u32, y as u32, z as u32, new_val);
                }
            }
        }
    }
}

/// Marker: this chunk needs remeshing
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct DensityFieldDirty;

/// Marker: chunk is currently being meshed on GPU
#[derive(Component)]
pub struct MeshingInProgress;

/// Cached neighbor field data for seamless meshing
#[derive(Component, Clone, Debug, Default)]
pub struct NeighborDensityFields {
    pub neighbors: [Option<NeighborSlice>; 6],
}

#[derive(Clone, Debug)]
pub struct NeighborSlice {
    pub data: Vec<f32>,
}

impl NeighborSlice {
    pub fn from_field(field: &DensityField, face: NeighborFace) -> Self {
        let (size_a, size_b, get_idx) = match face {
            NeighborFace::NegX => (
                DENSITY_FIELD_SIZE.y,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a: u32, b: u32| DensityField::index(DENSITY_FIELD_SIZE.x - 1, a, b))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            NeighborFace::PosX => (
                DENSITY_FIELD_SIZE.y,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a: u32, b: u32| DensityField::index(0, a, b))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            NeighborFace::NegY => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a: u32, b: u32| DensityField::index(a, DENSITY_FIELD_SIZE.y - 1, b))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            NeighborFace::PosY => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a: u32, b: u32| DensityField::index(a, 0, b))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            NeighborFace::NegZ => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.y,
                Box::new(|a: u32, b: u32| DensityField::index(a, b, DENSITY_FIELD_SIZE.z - 1))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            NeighborFace::PosZ => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.y,
                Box::new(|a: u32, b: u32| DensityField::index(a, b, 0))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
        };

        let mut data = Vec::with_capacity((size_a * size_b) as usize);
        for b in 0..size_b {
            for a in 0..size_a {
                data.push(field.0[get_idx(a, b)]);
            }
        }

        Self { data }
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

    pub fn opposite(&self) -> Self {
        match self {
            Self::NegX => Self::PosX,
            Self::PosX => Self::NegX,
            Self::NegY => Self::PosY,
            Self::PosY => Self::NegY,
            Self::NegZ => Self::PosZ,
            Self::PosZ => Self::NegZ,
        }
    }
}

// ============================================================================
// CPU-side Surface Nets (for now, until GPU readback is working)
// ============================================================================

/// Generate mesh on CPU - this is the fallback/working implementation
fn generate_mesh_cpu(
    field: &DensityField,
    neighbors: &NeighborDensityFields,
    mesh_size: Vec3,
    chunk_offset: Vec3,
) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut vertex_lookup = vec![NULL_VERTEX; FIELD_VOLUME];

    let sample = |x: i32, y: i32, z: i32| -> f32 {
        if DensityField::in_bounds(x, y, z) {
            field.get(x as u32, y as u32, z as u32)
        } else {
            // Sample from neighbors
            if x < 0 {
                if let Some(ref slice) = neighbors.neighbors[NeighborFace::NegX as usize] {
                    let idx = (y as u32 + z as u32 * DENSITY_FIELD_SIZE.y) as usize;
                    if idx < slice.data.len() {
                        return slice.data[idx];
                    }
                }
            } else if x >= DENSITY_FIELD_SIZE.x as i32 {
                if let Some(ref slice) = neighbors.neighbors[NeighborFace::PosX as usize] {
                    let idx = (y as u32 + z as u32 * DENSITY_FIELD_SIZE.y) as usize;
                    if idx < slice.data.len() {
                        return slice.data[idx];
                    }
                }
            }
            if y < 0 {
                if let Some(ref slice) = neighbors.neighbors[NeighborFace::NegY as usize] {
                    let idx = (x as u32 + z as u32 * DENSITY_FIELD_SIZE.x) as usize;
                    if idx < slice.data.len() {
                        return slice.data[idx];
                    }
                }
            } else if y >= DENSITY_FIELD_SIZE.y as i32 {
                if let Some(ref slice) = neighbors.neighbors[NeighborFace::PosY as usize] {
                    let idx = (x as u32 + z as u32 * DENSITY_FIELD_SIZE.x) as usize;
                    if idx < slice.data.len() {
                        return slice.data[idx];
                    }
                }
            }
            if z < 0 {
                if let Some(ref slice) = neighbors.neighbors[NeighborFace::NegZ as usize] {
                    let idx = (x as u32 + y as u32 * DENSITY_FIELD_SIZE.x) as usize;
                    if idx < slice.data.len() {
                        return slice.data[idx];
                    }
                }
            } else if z >= DENSITY_FIELD_SIZE.z as i32 {
                if let Some(ref slice) = neighbors.neighbors[NeighborFace::PosZ as usize] {
                    let idx = (x as u32 + y as u32 * DENSITY_FIELD_SIZE.x) as usize;
                    if idx < slice.data.len() {
                        return slice.data[idx];
                    }
                }
            }
            1.0 // Outside
        }
    };

    // Cube corners
    const CORNERS: [[i32; 3]; 8] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
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

    let grid_to_world = |gx: f32, gy: f32, gz: f32| -> [f32; 3] {
        let scale = mesh_size / DENSITY_FIELD_SIZE.as_vec3();
        [
            gx * scale.x + chunk_offset.x,
            gy * scale.y + chunk_offset.y,
            gz * scale.z + chunk_offset.z,
        ]
    };

    // Pass 1: Generate vertices
    for z in 0..DENSITY_FIELD_SIZE.z {
        for y in 0..DENSITY_FIELD_SIZE.y {
            for x in 0..DENSITY_FIELD_SIZE.x {
                let ix = x as i32;
                let iy = y as i32;
                let iz = z as i32;

                // Sample 8 corners
                let mut corner_dists = [0.0f32; 8];
                let mut num_negative = 0;

                for (i, c) in CORNERS.iter().enumerate() {
                    corner_dists[i] = sample(ix + c[0], iy + c[1], iz + c[2]);
                    if corner_dists[i] < 0.0 {
                        num_negative += 1;
                    }
                }

                let stride = DensityField::index(x, y, z);

                // No surface crossing
                if num_negative == 0 || num_negative == 8 {
                    vertex_lookup[stride] = NULL_VERTEX;
                    continue;
                }

                // Calculate centroid of edge intersections
                let mut sum = [0.0f32; 3];
                let mut count = 0.0;

                for edge in &EDGES {
                    let d1 = corner_dists[edge[0]];
                    let d2 = corner_dists[edge[1]];

                    if (d1 < 0.0) != (d2 < 0.0) {
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

                let grid_pos = [
                    x as f32 + centroid[0],
                    y as f32 + centroid[1],
                    z as f32 + centroid[2],
                ];
                let world_pos = grid_to_world(grid_pos[0], grid_pos[1], grid_pos[2]);

                // Calculate normal via gradient
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

    // Pass 2: Generate indices
    let stride_x = 1u32;
    let stride_y = DENSITY_FIELD_SIZE.x;
    let stride_z = DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.y;

    for z in 0..DENSITY_FIELD_SIZE.z {
        for y in 0..DENSITY_FIELD_SIZE.y {
            for x in 0..DENSITY_FIELD_SIZE.x {
                let stride = DensityField::index(x, y, z);
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
                if y > 0 && z > 0 && (d0 < 0.0) != (dx < 0.0) {
                    let v1 = vertex_lookup[stride - stride_y as usize];
                    let v2 = vertex_lookup[stride - stride_z as usize];
                    let v3 = vertex_lookup[stride - stride_y as usize - stride_z as usize];

                    if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
                        if d0 < 0.0 {
                            indices.extend_from_slice(&[v0, v3, v1, v0, v2, v3]);
                        } else {
                            indices.extend_from_slice(&[v0, v1, v3, v0, v3, v2]);
                        }
                    }
                }

                // Y-axis edge
                if x > 0 && z > 0 && (d0 < 0.0) != (dy < 0.0) {
                    let v1 = vertex_lookup[stride - stride_z as usize];
                    let v2 = vertex_lookup[stride - stride_x as usize];
                    let v3 = vertex_lookup[stride - stride_x as usize - stride_z as usize];

                    if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
                        if d0 < 0.0 {
                            indices.extend_from_slice(&[v0, v3, v1, v0, v2, v3]);
                        } else {
                            indices.extend_from_slice(&[v0, v1, v3, v0, v3, v2]);
                        }
                    }
                }

                // Z-axis edge
                if x > 0 && y > 0 && (d0 < 0.0) != (dz < 0.0) {
                    let v1 = vertex_lookup[stride - stride_x as usize];
                    let v2 = vertex_lookup[stride - stride_y as usize];
                    let v3 = vertex_lookup[stride - stride_x as usize - stride_y as usize];

                    if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
                        if d0 < 0.0 {
                            indices.extend_from_slice(&[v0, v3, v1, v0, v2, v3]);
                        } else {
                            indices.extend_from_slice(&[v0, v1, v3, v0, v3, v2]);
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

// ============================================================================
// Main World Systems
// ============================================================================

/// Auto-mark chunks dirty when their density field changes
fn auto_mark_dirty(mut commands: Commands, changed: Query<Entity, Changed<DensityField>>) {
    for entity in changed.iter() {
        commands.entity(entity).insert(DensityFieldDirty);
    }
}

/// Gather neighbor density slices for dirty chunks
fn gather_neighbor_fields(
    mut commands: Commands,
    dirty_chunks: Query<(Entity, &ChunkPos), (With<DensityFieldDirty>, With<DensityField>)>,
    all_fields: Query<&DensityField>,
    chunk_manager: Res<ChunkManager>,
) {
    for (entity, chunk_pos) in dirty_chunks.iter() {
        let mut neighbors = NeighborDensityFields::default();

        for face in NeighborFace::ALL {
            let neighbor_pos = chunk_pos.0 + face.offset();

            if let Some(neighbor_entity) = chunk_manager.get_chunk(&neighbor_pos) {
                if let Ok(neighbor_field) = all_fields.get(neighbor_entity) {
                    neighbors.neighbors[face as usize] =
                        Some(NeighborSlice::from_field(neighbor_field, face.opposite()));
                }
            }
        }

        commands.entity(entity).insert(neighbors);
    }
}

/// Process dirty chunks and generate meshes (CPU version)
fn process_dirty_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    dirty_chunks: Query<
        (
            Entity,
            &ChunkPos,
            &DensityField,
            Option<&NeighborDensityFields>,
        ),
        With<DensityFieldDirty>,
    >,
    mesh_size: Res<DensityFieldMeshSize>,
    existing_meshes: Query<&Mesh3d>,
) {
    for (entity, chunk_pos, field, neighbors) in dirty_chunks.iter() {
        let neighbors = neighbors.cloned().unwrap_or_default();
        let chunk_offset = chunk_pos.as_vec3() * mesh_size.0;

        if let Some(mesh) = generate_mesh_cpu(field, &neighbors, mesh_size.0, chunk_offset) {
            let mesh_handle = meshes.add(mesh);

            // Check if entity already has a mesh
            if existing_meshes.get(entity).is_ok() {
                // Update existing mesh
                commands.entity(entity).insert(Mesh3d(mesh_handle));
            } else {
                // Add new mesh and material
                commands.entity(entity).insert((
                    Mesh3d(mesh_handle),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(0.5, 0.7, 0.5),
                        perceptual_roughness: 0.8,
                        ..default()
                    })),
                ));
            }
        }

        // Remove dirty flag
        commands.entity(entity).remove::<DensityFieldDirty>();
    }
}

/// Handle mesh readback (placeholder for GPU version)
fn handle_mesh_readback() {
    // This will be used when we implement GPU readback
}

// ============================================================================
// Render World (GPU Compute - kept for future optimization)
// ============================================================================

#[derive(Resource, Default)]
pub struct ExtractedFields {
    pub fields: Vec<ExtractedField>,
}

pub struct ExtractedField {
    pub entity: Entity,
    pub chunk_pos: IVec3,
    pub data: Vec<f32>,
    pub neighbors: NeighborDensityFields,
    pub mesh_size: Vec3,
}

#[derive(Resource, Default)]
pub struct SurfaceNetsPipeline {
    pub vertex_pass: Option<CachedComputePipelineId>,
    pub index_pass: Option<CachedComputePipelineId>,
    pub bind_group_layout: Option<BindGroupLayout>,
}

#[derive(Resource, Default)]
pub struct GpuBufferStore {
    pub buffers: Vec<GpuFieldBuffers>,
}

pub struct GpuFieldBuffers {
    pub entity: Entity,
    pub bind_group: BindGroup,
    // Buffers stored here for future GPU readback
    pub positions: Buffer,
    pub normals: Buffer,
    pub indices: Buffer,
    pub counters: Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub mesh_size: [f32; 3],
    pub _pad0: f32,
    pub field_size: [u32; 3],
    pub _pad1: u32,
    pub chunk_offset: [f32; 3],
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Counters {
    pub vertex_count: u32,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NeighborFlagsGpu {
    pub flags: u32,
    pub _pad: [u32; 3],
}

fn extract_fields(mut extracted: ResMut<ExtractedFields>) {
    extracted.fields.clear();
}

fn setup_pipeline(render_device: Res<RenderDevice>, mut pipeline: ResMut<SurfaceNetsPipeline>) {
    // Pipeline setup kept for future GPU compute implementation
    let layout = render_device.create_bind_group_layout(
        Some("surface_nets_layout"),
        &[
            bgl_entry(0, BufferBindingType::Storage { read_only: true }),
            bgl_entry(1, BufferBindingType::Storage { read_only: true }),
            bgl_entry(2, BufferBindingType::Storage { read_only: true }),
            bgl_entry(3, BufferBindingType::Storage { read_only: true }),
            bgl_entry(4, BufferBindingType::Storage { read_only: true }),
            bgl_entry(5, BufferBindingType::Storage { read_only: true }),
            bgl_entry(6, BufferBindingType::Storage { read_only: true }),
            bgl_entry(7, BufferBindingType::Uniform),
            bgl_entry(8, BufferBindingType::Storage { read_only: false }),
            bgl_entry(9, BufferBindingType::Storage { read_only: false }),
            bgl_entry(10, BufferBindingType::Storage { read_only: false }),
            bgl_entry(11, BufferBindingType::Storage { read_only: false }),
            bgl_entry(12, BufferBindingType::Uniform),
            bgl_entry(13, BufferBindingType::Storage { read_only: false }),
        ],
    );

    pipeline.bind_group_layout = Some(layout);
}

fn bgl_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn prepare_buffers(// Placeholder - CPU handles meshing for now
) {
}

#[derive(Default)]
pub struct SurfaceNetsNode;

impl bevy::render::render_graph::Node for SurfaceNetsNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        _render_context: &mut bevy::render::renderer::RenderContext,
        _world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        // GPU compute disabled for now - CPU handles meshing
        Ok(())
    }
}
