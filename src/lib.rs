pub use crate::{
    mesher::DensityFieldMeshSize,
    neighbor::{NeighborFace, NeighborSlice},
    prelude::{DefaultIsoField, GenerateMesh, NeighborDensityFields},
};
use bevy::prelude::*;
use chunky_bevy::prelude::*;

/// Density field storage and SDF operations.
pub mod density_field;
pub mod field;
/// Sculpting brush functions for modifying density fields.
pub mod helpers;
/// Surface Nets mesh generation.
pub mod mesher;
/// Neighbor chunk data for seamless boundaries.
pub mod neighbor;

/// Common imports for working with bevy-sculpter.
pub mod prelude {
    pub use crate::{
        DefaultChunkManager,
        SurfaceNetsPlugin,
        density_field::{DefaultIsoField, GenerateMesh},
        field::Field,
        mesher::DensityFieldMeshSize,
        // Export generic neighbor types for reuse
        neighbor::{
            DensitySlice, NEIGHBOR_DEPTH, NeighborDensityFields, NeighborFace, NeighborFields,
            NeighborSlice,
        },
    };
}

pub struct SurfaceNetsPlugin;

impl Plugin for SurfaceNetsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ChunkyPlugin)
            .init_resource::<DensityFieldMeshSize>();

        #[cfg(feature = "auto-mesh")]
        app.add_systems(Update, auto_mark_generate);

        app.add_systems(Update, (process_chunks,).chain());
        app.register_sculpter_instance::<DefaultChunkManagerResource>();
    }
}

pub trait RegisterSufaceNetsChunkManager {
    fn register_sculpter_instance<T: ChunkManaging + Send + Sync + 'static>(&mut self)
    -> &mut Self;
}
impl RegisterSufaceNetsChunkManager for App {
    fn register_sculpter_instance<T: ChunkManaging + Send + Sync + 'static>(
        &mut self,
    ) -> &mut Self {
        self.add_systems(PreUpdate, gather_neighbor_fields::<T>)
            .init_resource::<ChunkManagerResource<T>>();
        self
    }
}

pub type DefaultChunkManager = ChunkManagerResource<DefaultChunkManagerResource>;

pub struct DefaultChunkManagerResource;
impl ChunkManaging for DefaultChunkManagerResource {
    const DIMENSIONS: Vec3 = Vec3::splat(10.0);
}

/// Auto-mark chunks for mesh generation when their density field changes
#[cfg(feature = "auto-mesh")]
fn auto_mark_generate(
    mut commands: Commands,
    changed: Query<Entity, (Changed<DefaultIsoField>, Without<GenerateMesh>)>,
) {
    for entity in changed.iter() {
        commands.entity(entity).insert(GenerateMesh);
    }
}

/// Gather neighbor density slices for chunks pending mesh generation
pub fn gather_neighbor_fields<T>(
    mut commands: Commands,
    pending_chunks: Query<(Entity, &ChunkPositon), (With<GenerateMesh>, With<DefaultIsoField>)>,
    all_fields: Query<&DefaultIsoField>,
    chunk_manager_resource: Res<ChunkManagerResource<T>>,
    chunk_managers: Query<&ChunkManager>,
) where
    T: ChunkManaging + Send + Sync + 'static,
{
    let Ok(chunk_manager) = chunk_managers.get(chunk_manager_resource.entity) else {
        return;
    };
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
        (Entity, &DefaultIsoField, Option<&NeighborDensityFields>),
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
