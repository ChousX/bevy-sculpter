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

use bevy::prelude::*;
use chunky_bevy::ChunkyPlugin;
pub use chunky_bevy::prelude::{Chunk, ChunkManager, ChunkPos};

pub use crate::{
    mesher::MeshSize,
    neighbor::{NeighborFace, NeighborIsoFields, NeighborSlice},
    prelude::{GenerateMesh, SdfVolume},
};

pub mod field;
pub mod helpers;
pub mod mesher;
pub mod neighbor;
pub mod sculptable;
pub mod sdf_volume;

/// Common imports for working with bevy-sculpter.
pub mod prelude {
    pub use crate::{
        FIELD_SIZE, SurfaceNetsExt, SurfaceNetsPlugin,
        field::Field,
        mesher::MeshSize,
        neighbor::{NEIGHBOR_DEPTH, NeighborFace, NeighborIsoFields, NeighborSlice},
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
///     .register_sculptable::<WaterField, f32>()
/// ```
pub struct SurfaceNetsPlugin;

impl Plugin for SurfaceNetsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ChunkyPlugin::default())
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
    /// * `T` - Storage type of the field
    ///
    /// # Features
    /// - With `auto-mesh` feature: automatically marks entities for remeshing when `F` changes
    ///
    /// # Example
    ///
    /// ```ignore
    /// app.register_sculptable::<SdfVolume, f32>()
    ///    .register_sculptable::<BinaryVoxelField, u8>();
    /// ```
    fn register_sculptable<F, T>(&mut self) -> &mut Self
    where
        F: sculptable::Sculptable<T> + Component,
        T: Copy + Default + Send + Sync + 'static;
}

impl SurfaceNetsExt for App {
    fn register_sculptable<F, T>(&mut self) -> &mut Self
    where
        F: sculptable::Sculptable<T> + Component,
        T: Copy + Default + Send + Sync + 'static,
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
                gather_neighbor_iso_fields::<F, T>,
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
    T: Copy + Default + Send + Sync + 'static,
{
    for entity in changed.iter() {
        commands.entity(entity).insert(GenerateMesh);
    }
}

/// Gather pre-converted neighbor iso slices for child entities pending mesh generation.
///
/// This system:
/// 1. Finds sculptable children with `GenerateMesh`
/// 2. Traverses `ChildOf` to get parent chunk's `ChunkPos`
/// 3. Gathers neighbor data from children of neighboring chunks with same `F` component
/// 4. Pre-converts all values to iso (f32) for efficient meshing
fn gather_neighbor_iso_fields<F, T>(
    mut commands: Commands,
    // Children that need meshing (no neighbor data yet)
    pending: Query<(Entity, &ChildOf), (With<F>, With<GenerateMesh>, Without<NeighborIsoFields>)>,
    // Parent chunks with position
    chunks: Query<&ChunkPos, With<Chunk>>,
    // All fields of this type (to find in neighbor chunks' children)
    all_fields: Query<&F>,
    // Chunk lookup
    chunk_manager: Res<ChunkManager>,
    // To find children of neighbor chunks
    children_query: Query<&Children>,
) where
    F: sculptable::Sculptable<T> + Component,
    T: Copy + Default + Send + Sync + 'static,
{
    for (entity, child_of) in pending.iter() {
        // Get parent chunk's position
        let Ok(chunk_pos) = chunks.get(child_of.parent()) else {
            continue;
        };

        let mut neighbors: [Option<NeighborSlice<f32>>; 6] = Default::default();

        for face in NeighborFace::ALL {
            let neighbor_pos = chunk_pos.0 + face.offset();

            // Find neighbor chunk entity
            let Some(neighbor_chunk) = chunk_manager.get_chunk(&neighbor_pos) else {
                continue;
            };

            // Find children of neighbor chunk
            let Ok(neighbor_children) = children_query.get(neighbor_chunk) else {
                continue;
            };

            // Find child with component F and extract boundary slice
            for child in neighbor_children.iter() {
                if let Ok(field) = all_fields.get(child) {
                    // Create pre-converted iso slice
                    neighbors[face as usize] =
                        Some(NeighborSlice::from_sampler(face, F::SIZE, |a, b, depth| {
                            let (x, y, z) = face.to_field_coords(a, b, depth, F::SIZE);
                            F::to_iso(field.get(x, y, z))
                        }));
                    break; // Only one field of type F per chunk
                }
            }
        }

        commands
            .entity(entity)
            .insert(NeighborIsoFields { neighbors });
    }
}

/// Generate meshes for sculptable children that have neighbor data ready.
///
/// The mesh is attached to the child entity, inheriting its `Transform`.
fn process_sculptable_mesh<F, T>(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    pending: Query<(Entity, &F, &NeighborIsoFields), With<GenerateMesh>>,
    mesh_size: Res<MeshSize>,
    existing_meshes: Query<&Mesh3d>,
) where
    F: sculptable::Sculptable<T> + Component,
    T: Copy + Default + Send + Sync + 'static,
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
            .remove::<NeighborIsoFields>();
    }
}

// ============================================================================
// Implement Sculptable for SdfVolume
// ============================================================================

impl sculptable::Sculptable<f32> for sdf_volume::SdfVolume {
    fn to_iso(value: f32) -> f32 {
        value // Identity - already a signed distance
    }
}

// ============================================================================
// Backward compatibility aliases
// ============================================================================

/// Alias for backward compatibility.
#[deprecated(since = "0.2.0", note = "Renamed to SdfVolume")]
pub type DensityField = sdf_volume::SdfVolume;

/// Alias for backward compatibility.
#[deprecated(since = "0.2.0", note = "Renamed to FIELD_SIZE")]
pub const DENSITY_FIELD_SIZE: UVec3 = FIELD_SIZE;

/// Alias for backward compatibility.
#[deprecated(since = "0.2.0", note = "Renamed to MeshSize")]
pub type DensityFieldMeshSize = MeshSize;
