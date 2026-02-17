//! # bevy-sculpter
//!
//! SDF-based voxel sculpting and Surface Nets meshing for Bevy.
//!
//! ## Quick Start
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_sculpter::prelude::*;
//! use chunky_bevy::prelude::*;
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(SurfaceNetsPlugin)
//!         .register_sculptable::<SdfVolume, f32>()
//!         .insert_resource(MeshSize(vec3(10., 10., 10.)))
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands) {
//!     // Spawn chunk parent
//!     let chunk = commands.spawn((Chunk, ChunkPos(ivec3(0, 0, 0)))).id();
//!     
//!     // Spawn sculptable field as child - mesh inherits child's transform
//!     let mut field = SdfVolume::new();
//!     bevy_sculpter::helpers::fill_centered_sphere(&mut field, 12.0);
//!     
//!     commands.spawn((
//!         field,
//!         GenerateMesh,
//!         Transform::default(),
//!     )).set_parent(chunk);
//! }
//! ```
//!
//! ## GenerateMesh Behavior
//!
//! - **On child entity**: Only that child's mesh is regenerated
//! - **On parent chunk**: Propagates to ALL children, regenerating all meshes

#![warn(missing_docs)]

pub use crate::{
    mesher::MeshSize,
    neighbor::{NeighborFace, NeighborFields, NeighborSlice},
    prelude::{GenerateMesh, SdfVolume},
};
use bevy::prelude::*;
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
        FIELD_SIZE, FIELD_VOLUME, SurfaceNetsExt, SurfaceNetsPlugin,
        field::Field,
        field_csg::{CsgOp, FieldCsg, IsoConvertible},
        mesher::MeshSize,
        neighbor::{NEIGHBOR_DEPTH, NeighborFace, NeighborFields, NeighborSlice},
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
            // Propagate GenerateMesh from parent chunks to all children
            .add_systems(Update, propagate_generate_mesh_to_children);
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
        F: sculptable::Sculptable<T> + Component,
        T: IsoConvertible + Send + Sync + 'static;
}

impl SurfaceNetsExt for App {
    fn register_sculptable<F, T>(&mut self) -> &mut Self
    where
        F: sculptable::Sculptable<T> + Component,
        T: IsoConvertible + Send + Sync + 'static,
    {
        // Auto-mesh: mark changed fields for regeneration
        #[cfg(feature = "auto-mesh")]
        self.add_systems(
            Update,
            auto_mark_changed::<F, T>.before(propagate_generate_mesh_to_children),
        );

        // Mesh generation pipeline
        self.add_systems(
            Update,
            (
                gather_neighbor_fields::<F, T>,
                process_sculptable_mesh::<F, T>,
            )
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
/// When a neighbor chunk exists but has no child with component `F`, the neighbor
/// slice is filled with `T::from_iso(1.0)` (outside) so boundary faces generate correctly.
fn gather_neighbor_fields<F, T>(
    mut commands: Commands,
    pending: Query<(Entity, &ChildOf), (With<F>, With<GenerateMesh>, Without<NeighborFields<T>>)>,
    chunks: Query<&ChunkPosition, With<Chunk>>,
    all_fields: Query<&F>,
    chunk_manager: Res<ChunkManager>,
    children_query: Query<&Children>,
) where
    F: sculptable::Sculptable<T> + Component,
    T: IsoConvertible + Send + Sync + 'static,
{
    for (entity, child_of) in pending.iter() {
        let Ok(chunk_pos) = chunks.get(child_of.parent()) else {
            continue;
        };

        let mut neighbors: [Option<NeighborSlice<T>>; 6] = Default::default();

        for face in NeighborFace::ALL {
            let neighbor_pos = chunk_pos.0 + face.offset();

            let Some(neighbor_chunk) = chunk_manager.get_chunk(&neighbor_pos) else {
                continue;
            };

            let Ok(neighbor_children) = children_query.get(neighbor_chunk) else {
                // Chunk exists but has no children — treat as outside
                neighbors[face as usize] =
                    Some(NeighborSlice::from_sampler(face, F::SIZE, |_, _, _| {
                        T::from_iso(1.0)
                    }));
                continue;
            };

            let mut found = false;
            for child in neighbor_children.iter() {
                if let Ok(field) = all_fields.get(child) {
                    neighbors[face as usize] = Some(NeighborSlice::from_field(field, face));
                    found = true;
                    break;
                }
            }

            // Chunk has children but none with component F — treat as outside
            if !found {
                neighbors[face as usize] =
                    Some(NeighborSlice::from_sampler(face, F::SIZE, |_, _, _| {
                        T::from_iso(1.0)
                    }));
            }
        }

        commands.entity(entity).insert(NeighborFields { neighbors });
    }
}

/// Generate meshes for sculptable children that have neighbor data ready.
///
/// The mesh is attached to the child entity, inheriting its `Transform`.
fn process_sculptable_mesh<F, T>(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    pending: Query<(Entity, &F, &NeighborFields<T>), With<GenerateMesh>>,
    mesh_size: Res<MeshSize>,
    existing_meshes: Query<&Mesh3d>,
) where
    F: sculptable::Sculptable<T> + Component,
    T: IsoConvertible + Send + Sync + 'static,
{
    for (entity, field, neighbors) in pending.iter() {
        if let Some(mesh) = mesher::generate_mesh_cpu::<F, T>(field, neighbors, mesh_size.0) {
            let mesh_handle = meshes.add(mesh);

            if existing_meshes.get(entity).is_ok() {
                // Update existing mesh
                commands.entity(entity).insert(Mesh3d(mesh_handle));
            } else {
                // Add mesh + default material
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

        // Clean up markers
        commands
            .entity(entity)
            .remove::<GenerateMesh>()
            .remove::<NeighborFields<T>>();
    }
}

// ============================================================================
// Implement Sculptable for SdfVolume
// ============================================================================

impl sculptable::Sculptable<f32> for sdf_volume::SdfVolume {}

// ============================================================================
// Backward compatibility aliases
// ============================================================================

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
