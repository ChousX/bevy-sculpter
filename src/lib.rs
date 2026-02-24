pub use crate::{
    mesher::MeshSize,
    neighbor::{
        NeighborCorner, NeighborCornerSlice, NeighborEdge, NeighborEdgeSlice, NeighborFace,
        NeighborFields, NeighborSlice,
    },
    prelude::{GenerateMesh, SdfVolume},
};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on};
use chunky_bevy::ChunkyPlugin;
pub use chunky_bevy::prelude::{Chunk, ChunkManager, ChunkPosition};

use field_csg::IsoConvertible;

pub mod field;
pub mod field_csg;
pub mod helpers;
pub mod mesher;
pub mod neighbor;
pub mod sculptable;
pub mod sdf_volume;

/// Common imports for working with bevy-sculpter.
pub mod prelude {
    pub use crate::backwars_compatibility::*;
    pub use crate::{
        FIELD_SIZE, FIELD_VOLUME, LodLevel, SurfaceNetsExt, SurfaceNetsPlugin,
        field::Field,
        field_csg::{CsgOp, FieldCsg, IsoConvertible},
        mesher::MeshSize,
        neighbor::{
            NEIGHBOR_DEPTH, NeighborCorner, NeighborCornerSlice, NeighborEdge, NeighborEdgeSlice,
            NeighborFace, NeighborFields, NeighborSlice,
        },
        sculptable::{Sculptable, SdfOps},
        sdf_volume::{GenerateMesh, SdfVolume},
    };
}

/// Size of the field grid per chunk (32×32×32 voxels).
pub const FIELD_SIZE: UVec3 = uvec3(32, 32, 32);

/// Total number of voxels per chunk.
pub const FIELD_VOLUME: usize = (FIELD_SIZE.x * FIELD_SIZE.y * FIELD_SIZE.z) as usize;

/// Sentinel value indicating no vertex exists at a position.
pub const NULL_VERTEX: u32 = u32::MAX;

/// Level of detail for mesh generation.
///
/// Controls the sampling stride through the SDF field. The step size
/// is `2^level`, so:
/// - Level 0 = step 1 (full resolution, 32³)
/// - Level 1 = step 2 (half resolution, 16³)
/// - Level 2 = step 4 (quarter resolution, 8³)
/// - Level 3 = step 8 (eighth resolution, 4³)
///
/// Place this on a **chunk entity** (parent). When mesh generation is
/// triggered, the LOD level propagates to child field entities automatically.
///
/// If absent, defaults to level 0 (full resolution).
///
/// # Example
///
/// ```ignore
/// commands.spawn((
///     Chunk,
///     ChunkPosition(ivec3(0, 0, 0)),
///     LodLevel(2), // quarter resolution
/// ));
/// ```
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LodLevel(pub u8);

impl LodLevel {
    /// Returns the sampling step size: `2^level`.
    #[inline]
    pub fn step(&self) -> u32 {
        1 << self.0
    }
}

/// Marker component: entity always meshes at full resolution, ignoring parent LodLevel.
///
/// Place this on a **child field entity** (not the chunk). When mesh generation
/// runs, fields with this marker will use step 1 regardless of the chunk's LOD.
///
/// # Example
///
/// ```ignore
/// commands.spawn((DomainField::new(), IgnoreLod));
/// ```
#[derive(Component, Default, Clone, Copy, Debug)]
pub struct IgnoreLod;

/// Plugin that enables Surface Nets mesh generation.
///
/// This plugin sets up the core infrastructure including `GenerateMesh` propagation
/// from parent chunks to children. You must also register each sculptable type:
///
/// ```ignore
/// App::new()
///     .add_plugins(SurfaceNetsPlugin)
///     .register_sculptable::<SdfVolume, f32>()
///     .register_sculptable::<BinaryField, bool>()
/// ```
pub struct SurfaceNetsPlugin;

impl Plugin for SurfaceNetsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ChunkyPlugin)
            .init_resource::<MeshSize>()
            .init_resource::<MeshBudget>()
            .add_systems(Update, propagate_generate_mesh_to_children)
            .add_systems(Update, receive_mesh_results);
    }
}

/// Propagate `GenerateMesh` from parent chunks to all children.
///
/// This allows marking a chunk for regeneration, which will trigger
/// mesh generation for all sculptable children.
fn propagate_generate_mesh_to_children(
    mut commands: Commands,
    chunks: Query<(Entity, &Children), (With<Chunk>, With<GenerateMesh>)>,
) {
    for (chunk_entity, children) in chunks.iter() {
        for child in children.iter() {
            commands.entity(child).insert(GenerateMesh);
        }
        // Remove from parent after propagation
        commands.entity(chunk_entity).remove::<GenerateMesh>();
    }
}

/// Marker for an in-flight async mesh generation task.
///
/// Uses `Option` so we can `.take()` the task to avoid double-polling
/// while waiting for deferred command removal.
#[derive(Component)]
struct MeshTaskPending(Option<Task<Option<Mesh>>>);

/// Controls how many mesh tasks can be dispatched per frame.
/// Prevents frame spikes when many chunks need remeshing simultaneously.
#[derive(Resource, Clone, Copy)]
pub struct MeshBudget {
    /// Max new mesh tasks to spawn per frame
    pub max_dispatches_per_frame: usize,
}

impl Default for MeshBudget {
    fn default() -> Self {
        Self {
            max_dispatches_per_frame: 4,
        }
    }
}

/// Extension trait for registering sculptable field types.
///
/// Each registered type gets its own systems for:
/// - (Optional) Auto-marking changed fields for remeshing
/// - Gathering neighbor data from sibling chunk children
/// - Generating meshes via Surface Nets
pub trait SurfaceNetsExt {
    /// Register a sculptable field type for automatic mesh generation.
    ///
    /// # Type Parameters
    /// * `F` - Field type implementing `Sculptable<T>` and `Component`
    /// * `T` - Storage type of the field (e.g., `f32`, `bool`, `u8`)
    ///
    /// # Features
    /// - With `auto-mesh` feature: automatically marks entities for remeshing when `F` changes
    ///
    /// # Example
    ///
    /// ```ignore
    /// app.register_sculptable::<SdfVolume, f32>()
    ///    .register_sculptable::<BinaryVoxelField, bool>();
    /// ```
    fn register_sculptable<F, T>(&mut self) -> &mut Self
    where
        F: sculptable::Sculptable<T> + Component + Clone,
        T: IsoConvertible + Send + Sync + Clone + 'static;
}

impl SurfaceNetsExt for App {
    fn register_sculptable<F, T>(&mut self) -> &mut Self
    where
        F: sculptable::Sculptable<T> + Component + Clone + Send + Sync + 'static,
        T: IsoConvertible + Copy + Default + Send + Sync + 'static,
    {
        #[cfg(feature = "auto-mesh")]
        self.add_systems(
            Update,
            auto_mark_changed::<F, T>.before(propagate_generate_mesh_to_children),
        );

        // Mesh generation pipeline: gather → dispatch (async)
        self.add_systems(
            Update,
            (gather_neighbor_fields::<F, T>, dispatch_mesh_tasks::<F, T>)
                .chain()
                .after(propagate_generate_mesh_to_children),
        );

        self
    }
}

/// Auto-mark entities for remeshing when their sculptable field changes.
#[cfg(feature = "auto-mesh")]
fn auto_mark_changed<F, T>(
    mut commands: Commands,
    changed: Query<Entity, (Changed<F>, Without<GenerateMesh>)>,
) where
    F: sculptable::Sculptable<T> + Component,
    T: IsoConvertible + Send + Sync + 'static,
{
    for entity in changed.iter() {
        commands.entity(entity).insert(GenerateMesh);
    }
}

/// Gather raw neighbor field values for child entities pending mesh generation.
///
/// This system:
/// 1. Finds sculptable children with `GenerateMesh`
/// 2. Traverses `ChildOf` to get parent chunk's `ChunkPos`
/// 3. Gathers neighbor data from children of neighboring chunks with same `F` component
/// 4. Stores raw `T` values (conversion to iso happens during meshing)
///
/// Gathers all 26 neighbors: 6 faces, 12 edges, 8 corners.
fn gather_neighbor_fields<F, T>(
    mut commands: Commands,
    // Children that need meshing (no neighbor data yet)
    pending: Query<(Entity, &ChildOf), (With<F>, With<GenerateMesh>, Without<NeighborFields<T>>)>,
    // Parent chunks with position
    chunks: Query<&ChunkPosition, With<Chunk>>,
    // All fields of this type (to find in neighbor chunks' children)
    all_fields: Query<&F>,
    // Chunk lookup
    chunk_manager: Res<ChunkManager>,
    // To find children of neighbor chunks
    children_query: Query<&Children>,
) where
    F: sculptable::Sculptable<T> + Component,
    T: IsoConvertible + Send + Sync + 'static,
{
    for (entity, child_of) in pending.iter() {
        // Get parent chunk's position
        let Ok(chunk_pos) = chunks.get(child_of.parent()) else {
            continue;
        };

        let mut neighbor_fields = NeighborFields::<T>::default();

        // Helper: find the field component F in a chunk's children
        let find_field = |pos: IVec3| -> Option<Entity> {
            let chunk = chunk_manager.get_chunk(&pos)?;
            let children = children_query.get(chunk).ok()?;
            children
                .iter()
                .find(|child| all_fields.get(*child).is_ok())
                .take()
        };

        // 6 face neighbors
        for face in NeighborFace::ALL {
            let neighbor_pos = chunk_pos.0 + face.offset();
            if let Some(child) = find_field(neighbor_pos) {
                if let Ok(field) = all_fields.get(child) {
                    neighbor_fields.neighbors[face as usize] =
                        Some(NeighborSlice::from_field(field, face));
                }
            }
        }

        // 12 edge neighbors
        for edge in NeighborEdge::ALL {
            let neighbor_pos = chunk_pos.0 + edge.offset();
            if let Some(child) = find_field(neighbor_pos) {
                if let Ok(field) = all_fields.get(child) {
                    neighbor_fields.edges[edge as usize] =
                        Some(NeighborEdgeSlice::from_field(field, edge));
                }
            }
        }

        // 8 corner neighbors
        for corner in NeighborCorner::ALL {
            let neighbor_pos = chunk_pos.0 + corner.offset();
            if let Some(child) = find_field(neighbor_pos) {
                if let Ok(field) = all_fields.get(child) {
                    neighbor_fields.corners[corner as usize] =
                        Some(NeighborCornerSlice::from_field(field, corner));
                }
            }
        }

        commands.entity(entity).insert(neighbor_fields);
    }
}

/// Dispatch mesh generation to the async compute thread pool.
///
/// Clones field + neighbor data, spawns a background task, and attaches
/// a `MeshTaskPending` component. The entity keeps its field data intact.
///
/// Reads `LodLevel` from the parent chunk to determine the sampling step.
/// If no `LodLevel` is present, defaults to full resolution (step 1).
fn dispatch_mesh_tasks<F, T>(
    mut commands: Commands,
    pending: Query<
        (Entity, &F, &NeighborFields<T>, &ChildOf, Has<IgnoreLod>),
        (With<GenerateMesh>, Without<MeshTaskPending>),
    >,
    chunk_lods: Query<Option<&LodLevel>, With<Chunk>>,
    mesh_size: Res<MeshSize>,
    budget: Res<MeshBudget>,
) where
    F: sculptable::Sculptable<T> + Component + Clone + Send + Sync + 'static,
    T: IsoConvertible + Copy + Default + Send + Sync + 'static,
{
    let pool = AsyncComputeTaskPool::get();
    let ms = mesh_size.0;

    let mut dispatched = 0;
    for (entity, field, neighbors, child_of, ignore_lod) in pending.iter() {
        if dispatched >= budget.max_dispatches_per_frame {
            break;
        }

        let step = if ignore_lod {
            1
        } else {
            chunk_lods
                .get(child_of.parent())
                .ok()
                .flatten()
                .map(|lod| lod.step())
                .unwrap_or(1)
        };

        let field_clone = field.clone();
        let neighbors_clone = neighbors.clone();

        let task = pool.spawn(async move {
            mesher::generate_mesh_cpu_lod::<F, T>(&field_clone, &neighbors_clone, ms, step)
        });

        commands
            .entity(entity)
            .remove::<GenerateMesh>()
            .remove::<NeighborFields<T>>()
            .insert(MeshTaskPending(Some(task)));

        dispatched += 1;
    }
}
/// Poll completed mesh tasks and insert the generated meshes.
fn receive_mesh_results(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut pending: Query<(Entity, &mut MeshTaskPending)>,
    existing_meshes: Query<&Mesh3d>,
) {
    for (entity, mut task) in pending.iter_mut() {
        // Already taken on a previous frame, waiting for deferred remove
        let Some(ref inner) = task.0 else {
            continue;
        };

        if !inner.is_finished() {
            continue;
        }

        // Take the task out so it can never be polled again
        let inner = task.0.take().unwrap();
        let result = block_on(inner);

        if let Some(mesh) = result {
            let mesh_handle = meshes.add(mesh);
            commands.entity(entity).insert(Mesh3d(mesh_handle));
        }

        commands.entity(entity).remove::<MeshTaskPending>();
    }
}

// Implement Sculptable for SdfVolume
impl sculptable::Sculptable<f32> for sdf_volume::SdfVolume {}

// Backward compatibility aliases

mod backwars_compatibility {
    /// Backward compatibility alias
    #[deprecated(since = "0.18.0", note = "Renamed to SdfVolume")]
    pub type DensityField = crate::SdfVolume;

    /// Backward compatibility alias
    #[deprecated(since = "0.18.0", note = "Renamed to FIELD_VOLUME")]
    pub const DENSITY_FIELD_VOLUME: usize = crate::FIELD_VOLUME;

    /// Backward compatibility alias
    #[deprecated(since = "0.18.0", note = "Renamed to FIELD_SIZE")]
    pub const DENSITY_FIELD_SIZE: bevy::math::UVec3 = crate::FIELD_SIZE;

    /// Backward compatibility alias
    #[deprecated(since = "0.18.0", note = "Renamed to MeshSize")]
    pub type DensityFieldMeshSize = crate::MeshSize;

    /// Backward compatibility alias
    #[deprecated(since = "0.2.0", note = "Renamed to GenerateMesh")]
    pub type DensityFieldDirty = crate::GenerateMesh;
}
