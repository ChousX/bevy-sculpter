// Surface Nets implementation inspired by fast-surface-nets-rs
// https://github.com/bonsairobo/fast-surface-nets-rs
// Original work Copyright 2021 bonsairobo, dual-licensed MIT/Apache-2.0
use bevy::prelude::*;
pub use chunky::prelude::{ChunkManager, ChunkPos};

use crate::{
    mesher::DensityFieldMeshSize,
    neighbor::{NeighborFace, NeighborSlice},
    prelude::{DensityField, DensityFieldDirty, NeighborDensityFields},
};

pub mod density_field;
pub mod mesher;
pub mod neighbor;

pub mod prelude {
    pub use crate::{
        DENSITY_FIELD_SIZE, SurfaceNetsPlugin,
        density_field::{DensityField, DensityFieldDirty},
        mesher::DensityFieldMeshSize,
        neighbor::NeighborDensityFields,
    };
}

/// Size of the density field grid per chunk (no padding)
pub const DENSITY_FIELD_SIZE: UVec3 = uvec3(32, 32, 32);

/// Total voxels per chunk
pub const FIELD_VOLUME: usize =
    (DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.y * DENSITY_FIELD_SIZE.z) as usize;

/// Null vertex marker
pub const NULL_VERTEX: u32 = u32::MAX;

pub struct SurfaceNetsPlugin;
impl Plugin for SurfaceNetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DensityFieldMeshSize>().add_systems(
            Update,
            (
                auto_mark_dirty,
                gather_neighbor_fields,
                process_dirty_chunks,
            )
                .chain(),
        );
    }
}

/// Auto-mark chunks dirty when their density field changes
fn auto_mark_dirty(
    mut commands: Commands,
    changed: Query<Entity, (Changed<DensityField>, Without<DensityFieldDirty>)>,
) {
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
                    // Get the boundary slice from the neighbor
                    neighbors.neighbors[face as usize] =
                        Some(NeighborSlice::from_field(neighbor_field, face));
                }
            }
        }

        commands.entity(entity).insert(neighbors);
    }
}

/// Process dirty chunks and generate meshes
fn process_dirty_chunks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    dirty_chunks: Query<
        (Entity, &DensityField, Option<&NeighborDensityFields>),
        With<DensityFieldDirty>,
    >,
    mesh_size: Res<DensityFieldMeshSize>,
    existing_meshes: Query<&Mesh3d>,
) {
    for (entity, field, neighbors) in dirty_chunks.iter() {
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

        commands.entity(entity).remove::<DensityFieldDirty>();
    }
}
