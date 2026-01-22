use crate::prelude::Sculptable;
pub use crate::{
    mesher::DensityFieldMeshSize,
    neighbor::{NeighborFace, NeighborSlice},
    prelude::{DefaultIsoField, GenerateMesh, NeighborDensityFields},
};
use bevy::prelude::*;
use chunky_bevy::prelude::*;

pub mod density_field;
pub mod field;
pub mod helpers;
pub mod mesher;
pub mod neighbor;
pub mod sculptable;

pub mod prelude {
    pub use crate::{
        DefaultChunkManager, RegisterSculptableField, SurfaceNetsPlugin,
        density_field::{DefaultIsoField, GenerateMesh},
        field::Field,
        mesher::DensityFieldMeshSize,
        neighbor::{
            DensitySlice, NEIGHBOR_DEPTH, NeighborDensityFields, NeighborFace, NeighborFields,
            NeighborSlice,
        },
        sculptable::Sculptable,
    };
}

pub struct SurfaceNetsPlugin;

impl Plugin for SurfaceNetsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ChunkyPlugin)
            .init_resource::<DensityFieldMeshSize>();

        // Register the default field type
        app.register_sculpter_instance::<DefaultChunkManagerResource>()
            .register_sculptable_field::<DefaultIsoField, DefaultChunkManagerResource>();
    }
}

// =============================================================================
// Chunk Manager Registration
// =============================================================================

pub trait RegisterSufaceNetsChunkManager {
    fn register_sculpter_instance<T: ChunkManaging>(&mut self) -> &mut Self;
}

impl RegisterSufaceNetsChunkManager for App {
    fn register_sculpter_instance<T: ChunkManaging>(&mut self) -> &mut Self {
        self.init_resource::<ChunkManagerResource<T>>();
        self
    }
}

// =============================================================================
// Sculptable Field Registration
// =============================================================================

/// Register a sculptable field type for automatic meshing.
///
/// This sets up the systems needed for:
/// - Auto-marking changed fields for remeshing (if `auto-mesh` feature enabled)
/// - Gathering neighbor data for seamless boundaries
/// - Generating meshes from the field
pub trait RegisterSculptableField {
    fn register_sculptable_field<F, CM>(&mut self) -> &mut Self
    where
        F: Sculptable<f32>,
        CM: ChunkManaging;
}

impl RegisterSculptableField for App {
    fn register_sculptable_field<F, CM>(&mut self) -> &mut Self
    where
        F: sculptable::Sculptable<f32>,
        CM: ChunkManaging,
    {
        #[cfg(feature = "auto-mesh")]
        self.add_systems(Update, auto_mark_generate::<F>);

        self.add_systems(
            PreUpdate,
            gather_neighbor_fields::<F, CM>.run_if(resource_exists::<ChunkManagerResource<CM>>),
        )
        .add_systems(Update, process_chunks::<F>);

        self
    }
}

// Default Chunk Manager

pub type DefaultChunkManager = ChunkManagerResource<DefaultChunkManagerResource>;

pub struct DefaultChunkManagerResource;

impl ChunkManaging for DefaultChunkManagerResource {
    const DIMENSIONS: Vec3 = Vec3::splat(10.0);
}

/// Auto-mark chunks for mesh generation when their field changes.
#[cfg(feature = "auto-mesh")]
fn auto_mark_generate<F: Component>(
    mut commands: Commands,
    changed: Query<Entity, (Changed<F>, Without<GenerateMesh>)>,
) {
    for entity in changed.iter() {
        commands.entity(entity).insert(GenerateMesh);
    }
}

/// Gather neighbor slices for chunks pending mesh generation.
pub fn gather_neighbor_fields<F, CM>(
    mut commands: Commands,
    pending_chunks: Query<(Entity, &ChunkPosition), (With<GenerateMesh>, With<F>)>,
    all_fields: Query<&F>,
    chunk_manager_resource: Res<ChunkManagerResource<CM>>,
    chunk_managers: Query<&ChunkManager>,
) where
    F: field::Field<f32> + Component,
    CM: ChunkManaging + Send + Sync + 'static,
{
    let Ok(chunk_manager) = chunk_managers.get(chunk_manager_resource.entity) else {
        return;
    };

    for (entity, chunk_pos) in pending_chunks.iter() {
        let neighbors = neighbor::NeighborFields::gather(|face| {
            let neighbor_pos = chunk_pos.0 + face.offset();
            chunk_manager
                .get_chunk(&neighbor_pos)
                .and_then(|e| all_fields.get(e).ok())
        });

        commands.entity(entity).insert(neighbors);
    }
}

/// Process chunks pending mesh generation.
pub fn process_chunks<F>(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    pending_chunks: Query<(Entity, &F, Option<&neighbor::NeighborFields<f32>>), With<GenerateMesh>>,
    mesh_size: Res<DensityFieldMeshSize>,
    existing_meshes: Query<&Mesh3d>,
) where
    F: sculptable::Sculptable<f32> + Component,
{
    for (entity, field, neighbors) in pending_chunks.iter() {
        let neighbors = neighbors.cloned().unwrap_or_default();

        if let Some(mesh) = mesher::generate_mesh(field, &neighbors, mesh_size.0) {
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
