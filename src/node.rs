use bevy::{
    prelude::*,
    render::{
        render_graph,
        render_resource::{ComputePassDescriptor, PipelineCache},
        renderer::RenderContext,
    },
};

use crate::{
    bind_group::SurfaceNetsBindGroups, buffers::SurfaceNetsBuffers, pipeline::SurfaceNetsPipelines,
};

const WORKGROUP_SIZE: u32 = 8;

#[derive(Default)]
pub struct SurfaceNetsNode;

impl render_graph::Node for SurfaceNetsNode {
    fn run<'w>(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> std::result::Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines = world.resource::<SurfaceNetsPipelines>();

        // Query all entities with both buffers and bind groups ready
        let mut query = world
            .try_query::<(&SurfaceNetsBuffers, &SurfaceNetsBindGroups)>()
            .unwrap();

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("surface_nets_compute_pass"),
                    ..default()
                });

        // Process each entity
        for (buffers, bind_groups) in query.iter(world) {
            // Calculate workgroup counts for this entity's dimensions
            let dims = buffers.dimensions.0;
            let workgroup_count_3d = uvec3(
                (dims.x + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                (dims.y + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                (dims.z + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
            );
            let cell_count = buffers.dimensions.cell_count();
            let workgroup_count_1d = (cell_count + 255) / 256;
            let max_faces = cell_count * 3;
            let face_workgroups = (max_faces + 255) / 256;
            // Stage 1: Generate Vertices
            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.generate_vertices_pipeline)
            {
                pass.set_bind_group(0, &bind_groups.generate_vertices, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(
                    workgroup_count_3d.x,
                    workgroup_count_3d.y,
                    workgroup_count_3d.z,
                );
            }

            // Stage 2: Prefix Sum (vertices)
            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.prefix_sum_pipeline)
            {
                pass.set_bind_group(0, &bind_groups.prefix_sum_vertices, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(workgroup_count_1d, 1, 1);
            }

            // Stage 3: Compact Vertices
            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.compact_vertices_pipeline)
            {
                pass.set_bind_group(0, &bind_groups.compact_vertices, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(workgroup_count_1d, 1, 1);
            }

            // Stage 4: Generate Faces
            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.generate_faces_pipeline)
            {
                pass.set_bind_group(0, &bind_groups.generate_faces, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(
                    workgroup_count_3d.x,
                    workgroup_count_3d.y,
                    workgroup_count_3d.z,
                );
            }

            // Stage 5: Prefix Sum (faces)
            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.prefix_sum_pipeline)
            {
                pass.set_bind_group(0, &bind_groups.prefix_sum_faces, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(face_workgroups, 1, 1);
            }

            // Stage 6: Compact Faces
            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.compact_faces_pipeline)
            {
                pass.set_bind_group(0, &bind_groups.compact_faces, &[]);
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(face_workgroups, 1, 1);
            }
        }
        pass.end();
        Ok(())
    }
}
