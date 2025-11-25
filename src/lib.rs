// src/density.rs
use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::{RenderGraph, RenderLabel},
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
};
use chunky::{Chunk, ChunkManager, ChunkPos};

pub mod prelude {
    pub use crate::{
        DensityField, DensityFieldDirty, DensityFieldMeshSize, NeighborDensityFields,
        SurfaceNetsPlugin,
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
            .add_plugins(ExtractComponentPlugin::<DensityFieldDirty>::default())
            .add_plugins(ExtractResourcePlugin::<DensityFieldMeshSize>::default())
            .add_systems(
                PostUpdate,
                (
                    auto_mark_dirty,
                    gather_neighbor_fields,
                    clear_dirty_after_extract,
                )
                    .chain(),
            );

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
#[derive(Component, ExtractComponent, Clone, Copy, Default, Debug)]
pub struct DensityFieldDirty;

/// Cached neighbor field data for seamless meshing
/// Order: -X, +X, -Y, +Y, -Z, +Z (6 faces)
/// Each contains only the boundary slice needed (1 voxel thick)
#[derive(Component, Clone, Debug, Default)]
pub struct NeighborDensityFields {
    /// Neighbor data: [neg_x, pos_x, neg_y, pos_y, neg_z, pos_z]
    /// Each is a flattened 2D slice of the neighboring chunk's boundary
    pub neighbors: [Option<NeighborSlice>; 6],
}

#[derive(Clone, Debug)]
pub struct NeighborSlice {
    /// Flattened boundary data (size depends on which face)
    pub data: Vec<f32>,
}

impl NeighborSlice {
    /// Create slice from neighbor's boundary
    pub fn from_field(field: &DensityField, face: NeighborFace) -> Self {
        let (size_a, size_b, get_idx) = match face {
            // -X face: sample x=DENSITY_FIELD_SIZE.x-1 from neighbor
            NeighborFace::NegX => (
                DENSITY_FIELD_SIZE.y,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a: u32, b: u32| DensityField::index(DENSITY_FIELD_SIZE.x - 1, a, b))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            // +X face: sample x=0 from neighbor
            NeighborFace::PosX => (
                DENSITY_FIELD_SIZE.y,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a: u32, b: u32| DensityField::index(0, a, b))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            // -Y face: sample y=DENSITY_FIELD_SIZE.y-1 from neighbor
            NeighborFace::NegY => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a: u32, b: u32| DensityField::index(a, DENSITY_FIELD_SIZE.y - 1, b))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            // +Y face: sample y=0 from neighbor
            NeighborFace::PosY => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.z,
                Box::new(|a: u32, b: u32| DensityField::index(a, 0, b))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            // -Z face: sample z=DENSITY_FIELD_SIZE.z-1 from neighbor
            NeighborFace::NegZ => (
                DENSITY_FIELD_SIZE.x,
                DENSITY_FIELD_SIZE.y,
                Box::new(|a: u32, b: u32| DensityField::index(a, b, DENSITY_FIELD_SIZE.z - 1))
                    as Box<dyn Fn(u32, u32) -> usize>,
            ),
            // +Z face: sample z=0 from neighbor
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

    /// Get value from slice
    #[inline]
    pub fn get(&self, a: u32, b: u32, size_a: u32) -> f32 {
        self.data[(a + b * size_a) as usize]
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
                    // Get the opposite face's boundary from neighbor
                    neighbors.neighbors[face as usize] =
                        Some(NeighborSlice::from_field(neighbor_field, face.opposite()));
                }
            }
        }

        commands.entity(entity).insert(neighbors);
    }
}

/// Clear dirty flag after extraction
fn clear_dirty_after_extract(
    mut commands: Commands,
    dirty: Query<Entity, With<DensityFieldDirty>>,
) {
    for entity in dirty.iter() {
        commands.entity(entity).remove::<DensityFieldDirty>();
    }
}

// ============================================================================
// Render World
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
    pub sdf: Buffer,
    pub neighbor_neg_x: Buffer,
    pub neighbor_pos_x: Buffer,
    pub neighbor_neg_y: Buffer,
    pub neighbor_pos_y: Buffer,
    pub neighbor_neg_z: Buffer,
    pub neighbor_pos_z: Buffer,
    pub neighbor_flags: Buffer, // Which neighbors are present
    pub positions: Buffer,
    pub normals: Buffer,
    pub indices: Buffer,
    pub counters: Buffer,
    pub uniforms: Buffer,
    pub vertex_lookup: Buffer,
    pub bind_group: BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub mesh_size: [f32; 3],
    pub _pad0: f32,
    pub field_size: [u32; 3],
    pub _pad1: u32,
    pub chunk_offset: [f32; 3], // World offset for this chunk
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
pub struct NeighborFlags {
    pub flags: u32, // Bitmask: bit 0 = neg_x, bit 1 = pos_x, etc.
    pub _pad: [u32; 3],
}

fn extract_fields(
    mut extracted: ResMut<ExtractedFields>,
    query: Query<
        (Entity, &ChunkPos, &DensityField, &NeighborDensityFields),
        With<DensityFieldDirty>,
    >,
    mesh_size: Res<DensityFieldMeshSize>,
) {
    extracted.fields.clear();
    for (entity, chunk_pos, field, neighbors) in query.iter() {
        extracted.fields.push(ExtractedField {
            entity,
            chunk_pos: chunk_pos.0,
            data: field.0.clone(),
            neighbors: neighbors.clone(),
            mesh_size: mesh_size.0,
        });
    }
}

fn setup_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    mut pipeline: ResMut<SurfaceNetsPipeline>,
) {
    let layout = render_device.create_bind_group_layout(
        Some("surface_nets_layout"),
        &[
            // 0: Main SDF
            bgl_entry(0, BufferBindingType::Storage { read_only: true }),
            // 1-6: Neighbor slices (neg_x, pos_x, neg_y, pos_y, neg_z, pos_z)
            bgl_entry(1, BufferBindingType::Storage { read_only: true }),
            bgl_entry(2, BufferBindingType::Storage { read_only: true }),
            bgl_entry(3, BufferBindingType::Storage { read_only: true }),
            bgl_entry(4, BufferBindingType::Storage { read_only: true }),
            bgl_entry(5, BufferBindingType::Storage { read_only: true }),
            bgl_entry(6, BufferBindingType::Storage { read_only: true }),
            // 7: Neighbor flags
            bgl_entry(7, BufferBindingType::Uniform),
            // 8: Positions output
            bgl_entry(8, BufferBindingType::Storage { read_only: false }),
            // 9: Normals output
            bgl_entry(9, BufferBindingType::Storage { read_only: false }),
            // 10: Indices output
            bgl_entry(10, BufferBindingType::Storage { read_only: false }),
            // 11: Counters
            bgl_entry(11, BufferBindingType::Storage { read_only: false }),
            // 12: Uniforms
            bgl_entry(12, BufferBindingType::Uniform),
            // 13: Vertex lookup
            bgl_entry(13, BufferBindingType::Storage { read_only: false }),
        ],
    );

    let shader = asset_server.load("shaders/surface_nets.wgsl");

    pipeline.vertex_pass = Some(
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("surface_nets_vertex".into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some("generate_vertices".into()),
            ..default()
        }),
    );

    pipeline.index_pass = Some(
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("surface_nets_index".into()),
            layout: vec![layout.clone()],
            shader,
            entry_point: Some("generate_indices".into()),
            ..default()
        }),
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

fn prepare_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<SurfaceNetsPipeline>,
    extracted: Res<ExtractedFields>,
    mut store: ResMut<GpuBufferStore>,
) {
    let Some(layout) = &pipeline.bind_group_layout else {
        return;
    };

    // Remove stale buffers
    let extracted_entities: Vec<_> = extracted.fields.iter().map(|f| f.entity).collect();
    store
        .buffers
        .retain(|b| extracted_entities.contains(&b.entity));

    for field in &extracted.fields {
        // Check if buffer exists
        if let Some(buf) = store.buffers.iter_mut().find(|b| b.entity == field.entity) {
            update_buffers(&render_queue, buf, field);
        } else {
            let buf = create_buffers(&render_device, &render_queue, layout, field);
            store.buffers.push(buf);
        }
    }
}

fn neighbor_slice_size(face: NeighborFace) -> usize {
    match face {
        NeighborFace::NegX | NeighborFace::PosX => {
            (DENSITY_FIELD_SIZE.y * DENSITY_FIELD_SIZE.z) as usize
        }
        NeighborFace::NegY | NeighborFace::PosY => {
            (DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.z) as usize
        }
        NeighborFace::NegZ | NeighborFace::PosZ => {
            (DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.y) as usize
        }
    }
}

fn create_buffers(
    device: &RenderDevice,
    queue: &RenderQueue,
    layout: &BindGroupLayout,
    field: &ExtractedField,
) -> GpuFieldBuffers {
    let sdf = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf"),
        contents: bytemuck::cast_slice(&field.data),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // Create neighbor buffers (with dummy data if neighbor doesn't exist)
    let create_neighbor = |face: NeighborFace| {
        let size = neighbor_slice_size(face);
        let data = field.neighbors.neighbors[face as usize]
            .as_ref()
            .map(|n| n.data.clone())
            .unwrap_or_else(|| vec![1.0; size]); // Default to "outside"

        device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(&format!("neighbor_{:?}", face)),
            contents: bytemuck::cast_slice(&data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        })
    };

    let neighbor_neg_x = create_neighbor(NeighborFace::NegX);
    let neighbor_pos_x = create_neighbor(NeighborFace::PosX);
    let neighbor_neg_y = create_neighbor(NeighborFace::NegY);
    let neighbor_pos_y = create_neighbor(NeighborFace::PosY);
    let neighbor_neg_z = create_neighbor(NeighborFace::NegZ);
    let neighbor_pos_z = create_neighbor(NeighborFace::PosZ);

    // Build flags bitmask
    let mut flags = 0u32;
    for (i, neighbor) in field.neighbors.neighbors.iter().enumerate() {
        if neighbor.is_some() {
            flags |= 1 << i;
        }
    }
    let neighbor_flags_data = NeighborFlags {
        flags,
        _pad: [0; 3],
    };
    let neighbor_flags = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("neighbor_flags"),
        contents: bytemuck::bytes_of(&neighbor_flags_data),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let positions = device.create_buffer(&BufferDescriptor {
        label: Some("positions"),
        size: (MAX_VERTICES * 3 * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let normals = device.create_buffer(&BufferDescriptor {
        label: Some("normals"),
        size: (MAX_VERTICES * 3 * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let indices = device.create_buffer(&BufferDescriptor {
        label: Some("indices"),
        size: (MAX_INDICES * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::INDEX | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let counters = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("counters"),
        contents: bytemuck::bytes_of(&Counters::default()),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    let uniforms_data = Uniforms {
        mesh_size: field.mesh_size.into(),
        _pad0: 0.0,
        field_size: [
            DENSITY_FIELD_SIZE.x,
            DENSITY_FIELD_SIZE.y,
            DENSITY_FIELD_SIZE.z,
        ],
        _pad1: 0,
        chunk_offset: (field.chunk_pos.as_vec3() * field.mesh_size).into(),
        _pad2: 0.0,
    };
    let uniforms = device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uniforms_data),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let vertex_lookup = device.create_buffer(&BufferDescriptor {
        label: Some("vertex_lookup"),
        size: (FIELD_VOLUME * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Init to NULL_VERTEX
    queue.write_buffer(
        &vertex_lookup,
        0,
        bytemuck::cast_slice(&vec![NULL_VERTEX; FIELD_VOLUME]),
    );

    let bind_group = device.create_bind_group(
        Some("surface_nets_bind_group"),
        layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: sdf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: neighbor_neg_x.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: neighbor_pos_x.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: neighbor_neg_y.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: neighbor_pos_y.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: neighbor_neg_z.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: neighbor_pos_z.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: neighbor_flags.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: positions.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 9,
                resource: normals.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 10,
                resource: indices.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 11,
                resource: counters.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 12,
                resource: uniforms.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 13,
                resource: vertex_lookup.as_entire_binding(),
            },
        ],
    );

    GpuFieldBuffers {
        entity: field.entity,
        sdf,
        neighbor_neg_x,
        neighbor_pos_x,
        neighbor_neg_y,
        neighbor_pos_y,
        neighbor_neg_z,
        neighbor_pos_z,
        neighbor_flags,
        positions,
        normals,
        indices,
        counters,
        uniforms,
        vertex_lookup,
        bind_group,
    }
}

fn update_buffers(queue: &RenderQueue, buf: &GpuFieldBuffers, field: &ExtractedField) {
    queue.write_buffer(&buf.sdf, 0, bytemuck::cast_slice(&field.data));

    // Update neighbors
    for face in NeighborFace::ALL {
        let size = neighbor_slice_size(face);
        let data = field.neighbors.neighbors[face as usize]
            .as_ref()
            .map(|n| n.data.clone())
            .unwrap_or_else(|| vec![1.0; size]);

        let buffer = match face {
            NeighborFace::NegX => &buf.neighbor_neg_x,
            NeighborFace::PosX => &buf.neighbor_pos_x,
            NeighborFace::NegY => &buf.neighbor_neg_y,
            NeighborFace::PosY => &buf.neighbor_pos_y,
            NeighborFace::NegZ => &buf.neighbor_neg_z,
            NeighborFace::PosZ => &buf.neighbor_pos_z,
        };
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&data));
    }

    // Update flags
    let mut flags = 0u32;
    for (i, neighbor) in field.neighbors.neighbors.iter().enumerate() {
        if neighbor.is_some() {
            flags |= 1 << i;
        }
    }
    queue.write_buffer(
        &buf.neighbor_flags,
        0,
        bytemuck::bytes_of(&NeighborFlags {
            flags,
            _pad: [0; 3],
        }),
    );

    // Reset counters
    queue.write_buffer(&buf.counters, 0, bytemuck::bytes_of(&Counters::default()));

    // Reset vertex lookup
    queue.write_buffer(
        &buf.vertex_lookup,
        0,
        bytemuck::cast_slice(&vec![NULL_VERTEX; FIELD_VOLUME]),
    );

    // Update uniforms
    let uniforms = Uniforms {
        mesh_size: field.mesh_size.into(),
        _pad0: 0.0,
        field_size: [
            DENSITY_FIELD_SIZE.x,
            DENSITY_FIELD_SIZE.y,
            DENSITY_FIELD_SIZE.z,
        ],
        _pad1: 0,
        chunk_offset: (field.chunk_pos.as_vec3() * field.mesh_size).into(),
        _pad2: 0.0,
    };
    queue.write_buffer(&buf.uniforms, 0, bytemuck::bytes_of(&uniforms));
}

// ============================================================================
// Compute Node
// ============================================================================

#[derive(Default)]
pub struct SurfaceNetsNode;

impl bevy::render::render_graph::Node for SurfaceNetsNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SurfaceNetsPipeline>();
        let store = world.resource::<GpuBufferStore>();

        let (Some(vertex_id), Some(index_id)) = (pipeline.vertex_pass, pipeline.index_pass) else {
            return Ok(());
        };

        let (Some(vertex_pipeline), Some(index_pipeline)) = (
            pipeline_cache.get_compute_pipeline(vertex_id),
            pipeline_cache.get_compute_pipeline(index_id),
        ) else {
            return Ok(());
        };

        let workgroups = (
            DENSITY_FIELD_SIZE.x.div_ceil(WORKGROUP_SIZE),
            DENSITY_FIELD_SIZE.y.div_ceil(WORKGROUP_SIZE),
            DENSITY_FIELD_SIZE.z.div_ceil(WORKGROUP_SIZE),
        );

        for buf in &store.buffers {
            // Pass 1: Generate vertices
            {
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("surface_nets_vertex"),
                            ..default()
                        });
                pass.set_pipeline(vertex_pipeline);
                pass.set_bind_group(0, &buf.bind_group, &[]);
                pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            }

            // Pass 2: Generate indices
            {
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("surface_nets_index"),
                            ..default()
                        });
                pass.set_pipeline(index_pipeline);
                pass.set_bind_group(0, &buf.bind_group, &[]);
                pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            }
        }

        Ok(())
    }
}
