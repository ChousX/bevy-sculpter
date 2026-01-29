//! # bevy-sculpter
//!
//! SDF-based voxel sculpting and Surface Nets meshing for Bevy.
//!
//! This crate provides tools for creating and manipulating volumetric data using
//! signed distance fields (SDFs), with automatic mesh generation via the Surface Nets
//! algorithm.
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
//!         .add_plugins(ChunkyPlugin::default())
//!         .add_plugins(SurfaceNetsPlugin::default())
//!         .insert_resource(DensityFieldMeshSize(vec3(10., 10., 10.)))
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands) {
//!     let mut field = DensityField::new();
//!     bevy_sculpter::helpers::fill_centered_sphere(&mut field, 12.0);
//!     
//!     commands.spawn((
//!         Chunk,
//!         ChunkPos(ivec3(0, 0, 0)),
//!         field,
//!         GenerateMesh,
//!     ));
//! }
//! ```
//!
//! ## Features
//!
//! - **[`DensityField`]**: SDF-based volumetric storage with raycasting and nearest-point queries
//! - **[`SurfaceNetsPlugin`]**: Automatic mesh generation with seamless chunk boundaries
//! - **[`helpers`]**: Sculpting brushes (smooth, hard, blur, flatten)
//! - **[`neighbor`]**: Generic neighbor data structures for seamless chunk boundaries
//!
//! ## Stability
//!
//! This crate is under active development. Breaking changes may occur between minor versions
//! until 1.0 is released.

// Surface Nets implementation inspired by fast-surface-nets-rs
// https://github.com/bonsairobo/fast-surface-nets-rs
// Original work Copyright 2021 bonsairobo, dual-licensed MIT/Apache-2.0

#![warn(missing_docs)]

use bevy::prelude::*;
use chunky_bevy::ChunkyPlugin;
pub use chunky_bevy::prelude::{ChunkManager, ChunkPos};

pub use crate::{
    mesher::DensityFieldMeshSize,
    neighbor::{NeighborFace, NeighborSlice},
    prelude::{DensityField, GenerateMesh, NeighborDensityFields},
};

/// Density field storage and SDF operations.
pub mod density_field;
pub mod field;
/// Sculpting brush functions for modifying density fields.
pub mod helpers;
/// Surface Nets mesh generation.
pub mod mesher;
/// Neighbor chunk data for seamless boundaries.
pub mod neighbor;
mod sculptable;

/// Common imports for working with bevy-sculpter.
pub mod prelude {
    pub use crate::{
        DENSITY_FIELD_SIZE,
        SurfaceNetsPlugin,
        density_field::{DensityField, GenerateMesh},
        field::Field,
        mesher::DensityFieldMeshSize,
        // Export generic neighbor types for reuse
        neighbor::{
            DensitySlice, NEIGHBOR_DEPTH, NeighborDensityFields, NeighborFace, NeighborFields,
            NeighborSlice,
        },
    };
}

/// Size of the density field grid per chunk (32×32×32 voxels).
pub const DENSITY_FIELD_SIZE: UVec3 = uvec3(32, 32, 32);

/// Total number of voxels per chunk.
pub const FIELD_VOLUME: usize =
    (DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.y * DENSITY_FIELD_SIZE.z) as usize;

/// Sentinel value indicating no vertex exists at a position.
pub const NULL_VERTEX: u32 = u32::MAX;

/// Plugin that enables automatic Surface Nets mesh generation for chunks with density fields.
///
/// When added to your app, this plugin will automatically generate and update meshes for any
/// entity that has both a [`DensityField`] and [`GenerateMesh`] components.
///
/// # Auto-marking
///
/// By default (with the `auto-mesh` feature enabled), chunks are automatically marked for
/// meshing when their [`DensityField`] changes. Disable this feature for manual control:
///
/// ```toml
/// bevy-sculpter = { version = "...", default-features = false }
/// ```
///
/// # Example
///
/// ```no_run
/// use bevy::prelude::*;
/// use bevy_sculpter::prelude::*;
/// use chunky_bevy::prelude::*;
///
/// App::new()
///     .add_plugins(DefaultPlugins)
///     .add_plugins(ChunkyPlugin::default())
///     .add_plugins(SurfaceNetsPlugin)
///     .insert_resource(DensityFieldMeshSize(vec3(10., 10., 10.)));
/// ```
pub struct SurfaceNetsPlugin;

impl Plugin for SurfaceNetsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ChunkyPlugin::default())
            .init_resource::<DensityFieldMeshSize>();

        #[cfg(feature = "auto-mesh")]
        app.add_systems(Update, auto_mark_generate);

        app.add_systems(Update, (gather_neighbor_fields, process_chunks).chain());
    }
}

/// Auto-mark chunks for mesh generation when their density field changes
#[cfg(feature = "auto-mesh")]
fn auto_mark_generate(
    mut commands: Commands,
    changed: Query<Entity, (Changed<DensityField>, Without<GenerateMesh>)>,
) {
    for entity in changed.iter() {
        commands.entity(entity).insert(GenerateMesh);
    }
}

/// Gather neighbor density slices for chunks pending mesh generation
fn gather_neighbor_fields(
    mut commands: Commands,
    pending_chunks: Query<(Entity, &ChunkPos), (With<GenerateMesh>, With<DensityField>)>,
    all_fields: Query<&DensityField>,
    chunk_manager: Res<ChunkManager>,
) {
    for (entity, chunk_pos) in pending_chunks.iter() {
        // Use the new gather method - much cleaner!
        let neighbors = NeighborDensityFields::gather(|face| {
            let neighbor_pos = chunk_pos.0 + face.offset();
            chunk_manager
                .get_chunk(&neighbor_pos)
                .and_then(|e| all_fields.get(e).ok())
        });

        commands.entity(entity).insert(neighbors);
    }
}

/// Process chunks pending mesh generation
fn process_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    pending_chunks: Query<
        (Entity, &DensityField, Option<&NeighborDensityFields>),
        With<GenerateMesh>,
    >,
    mesh_size: Res<DensityFieldMeshSize>,
    existing_meshes: Query<&Mesh3d>,
) {
    for (entity, field, neighbors) in pending_chunks.iter() {
        let neighbors = neighbors.cloned().unwrap_or_default();

        if let Some(mesh) = mesher::generate_mesh_cpu(field, &neighbors, mesh_size.0) {
            let mesh_handle = meshes.add(mesh);

            if existing_meshes.get(entity).is_ok() {
                commands.entity(entity).insert(Mesh3d(mesh_handle));
            } else {
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

        commands.entity(entity).remove::<GenerateMesh>();
    }
}
