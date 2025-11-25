// assets/shaders/surface_nets.wgsl

const FIELD_SIZE: vec3<u32> = vec3<u32>(32u, 32u, 32u);
const FIELD_VOLUME: u32 = 32768u; // 32^3

const MAX_VERTICES: u32 = 32768u;
const MAX_INDICES: u32 = 589824u;

const WORKGROUP_SIZE: u32 = 4u;
const NULL_VERTEX: u32 = 0xFFFFFFFFu;

// Neighbor flags
const HAS_NEG_X: u32 = 1u;
const HAS_POS_X: u32 = 2u;
const HAS_NEG_Y: u32 = 4u;
const HAS_POS_Y: u32 = 8u;
const HAS_NEG_Z: u32 = 16u;
const HAS_POS_Z: u32 = 32u;

// Bindings
@group(0) @binding(0) var<storage, read> sdf: array<f32>;

// Neighbor slices (boundary data from adjacent chunks)
@group(0) @binding(1) var<storage, read> neighbor_neg_x: array<f32>; // size: Y * Z
@group(0) @binding(2) var<storage, read> neighbor_pos_x: array<f32>; // size: Y * Z
@group(0) @binding(3) var<storage, read> neighbor_neg_y: array<f32>; // size: X * Z
@group(0) @binding(4) var<storage, read> neighbor_pos_y: array<f32>; // size: X * Z
@group(0) @binding(5) var<storage, read> neighbor_neg_z: array<f32>; // size: X * Y
@group(0) @binding(6) var<storage, read> neighbor_pos_z: array<f32>; // size: X * Y

struct NeighborFlags {
    flags: u32,
    _pad: vec3<u32>,
}
@group(0) @binding(7) var<uniform> neighbor_flags: NeighborFlags;

@group(0) @binding(8) var<storage, read_write> positions: array<f32>;
@group(0) @binding(9) var<storage, read_write> normals: array<f32>;
@group(0) @binding(10) var<storage, read_write> indices: array<u32>;

struct Counters {
    vertex_count: atomic<u32>,
    index_count: atomic<u32>,
}
@group(0) @binding(11) var<storage, read_write> counters: Counters;

struct Uniforms {
    mesh_size: vec3<f32>,
    _pad0: f32,
    field_size: vec3<u32>,
    _pad1: u32,
    chunk_offset: vec3<f32>,
    _pad2: f32,
}
@group(0) @binding(12) var<uniform> uniforms: Uniforms;

@group(0) @binding(13) var<storage, read_write> vertex_lookup: array<u32>;

// Cube data
const CUBE_CORNERS: array<vec3<u32>, 8> = array<vec3<u32>, 8>(
    vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u),
    vec3<u32>(0u, 1u, 0u), vec3<u32>(1u, 1u, 0u),
    vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u),
    vec3<u32>(0u, 1u, 1u), vec3<u32>(1u, 1u, 1u),
);

const CUBE_CORNER_VECTORS: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
);

const CUBE_EDGES: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2<u32>(0u, 1u), vec2<u32>(0u, 2u), vec2<u32>(0u, 4u),
    vec2<u32>(1u, 3u), vec2<u32>(1u, 5u), vec2<u32>(2u, 3u),
    vec2<u32>(2u, 6u), vec2<u32>(3u, 7u), vec2<u32>(4u, 5u),
    vec2<u32>(4u, 6u), vec2<u32>(5u, 7u), vec2<u32>(6u, 7u),
);

// ============================================================================
// Sampling functions with neighbor support
// ============================================================================

fn linearize(p: vec3<u32>) -> u32 {
    return p.x + p.y * FIELD_SIZE.x + p.z * FIELD_SIZE.x * FIELD_SIZE.y;
}

fn in_bounds(p: vec3<i32>) -> bool {
    return all(p >= vec3<i32>(0)) && all(vec3<u32>(p) < FIELD_SIZE);
}

/// Sample SDF with neighbor fallback for boundary voxels
fn sample_sdf(p: vec3<i32>) -> f32 {
    // In bounds - direct sample
    if in_bounds(p) {
        return sdf[linearize(vec3<u32>(p))];
    }

    // Out of bounds - check neighbors
    let flags = neighbor_flags.flags;

    // -X neighbor
    if p.x < 0 && (flags & HAS_NEG_X) != 0u {
        let local_y = u32(p.y);
        let local_z = u32(p.z);
        if local_y < FIELD_SIZE.y && local_z < FIELD_SIZE.z {
            return neighbor_neg_x[local_y + local_z * FIELD_SIZE.y];
        }
    }

    // +X neighbor
    if p.x >= i32(FIELD_SIZE.x) && (flags & HAS_POS_X) != 0u {
        let local_y = u32(p.y);
        let local_z = u32(p.z);
        if local_y < FIELD_SIZE.y && local_z < FIELD_SIZE.z {
            return neighbor_pos_x[local_y + local_z * FIELD_SIZE.y];
        }
    }

    // -Y neighbor
    if p.y < 0 && (flags & HAS_NEG_Y) != 0u {
        let local_x = u32(p.x);
        let local_z = u32(p.z);
        if local_x < FIELD_SIZE.x && local_z < FIELD_SIZE.z {
            return neighbor_neg_y[local_x + local_z * FIELD_SIZE.x];
        }
    }

    // +Y neighbor
    if p.y >= i32(FIELD_SIZE.y) && (flags & HAS_POS_Y) != 0u {
        let local_x = u32(p.x);
        let local_z = u32(p.z);
        if local_x < FIELD_SIZE.x && local_z < FIELD_SIZE.z {
            return neighbor_pos_y[local_x + local_z * FIELD_SIZE.x];
        }
    }

    // -Z neighbor
    if p.z < 0 && (flags & HAS_NEG_Z) != 0u {
        let local_x = u32(p.x);
        let local_y = u32(p.y);
        if local_x < FIELD_SIZE.x && local_y < FIELD_SIZE.y {
            return neighbor_neg_z[local_x + local_y * FIELD_SIZE.x];
        }
    }

    // +Z neighbor
    if p.z >= i32(FIELD_SIZE.z) && (flags & HAS_POS_Z) != 0u {
        let local_x = u32(p.x);
        let local_y = u32(p.y);
        if local_x < FIELD_SIZE.x && local_y < FIELD_SIZE.y {
            return neighbor_pos_z[local_x + local_y * FIELD_SIZE.x];
        }
    }

    // No neighbor available - return "outside"
    return 1.0;
}

// ============================================================================
// Surface nets core
// ============================================================================

fn edge_intersection(c1: u32, c2: u32, d1: f32, d2: f32) -> vec3<f32> {
    let t = d1 / (d1 - d2);
    return mix(CUBE_CORNER_VECTORS[c1], CUBE_CORNER_VECTORS[c2], t);
}

fn centroid_of_intersections(dists: array<f32, 8>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    var count = 0.0;

    for (var i = 0u; i < 12u; i++) {
        let edge = CUBE_EDGES[i];
        let d1 = dists[edge.x];
        let d2 = dists[edge.y];

        if (d1 < 0.0) != (d2 < 0.0) {
            sum += edge_intersection(edge.x, edge.y, d1, d2);
            count += 1.0;
        }
    }

    return select(vec3<f32>(0.5), sum / count, count > 0.0);
}

fn calculate_normal(dists: array<f32, 8>, s: vec3<f32>) -> vec3<f32> {
    let p00 = vec3<f32>(dists[1], dists[2], dists[4]);
    let n00 = vec3<f32>(dists[0], dists[0], dists[0]);
    let p10 = vec3<f32>(dists[5], dists[3], dists[6]);
    let n10 = vec3<f32>(dists[4], dists[1], dists[2]);
    let p01 = vec3<f32>(dists[3], dists[6], dists[5]);
    let n01 = vec3<f32>(dists[2], dists[4], dists[1]);
    let p11 = vec3<f32>(dists[7], dists[7], dists[7]);
    let n11 = vec3<f32>(dists[6], dists[5], dists[3]);

    let d00 = p00 - n00;
    let d10 = p10 - n10;
    let d01 = p01 - n01;
    let d11 = p11 - n11;

    let neg = vec3<f32>(1.0) - s;

    var grad = neg.yzx * neg.zxy * d00
             + neg.yzx * s.zxy * d10
             + s.yzx * neg.zxy * d01
             + s.yzx * s.zxy * d11;

    let len = length(grad);
    return select(vec3<f32>(0.0, 1.0, 0.0), grad / len, len > 0.0001);
}

fn grid_to_world(grid_pos: vec3<f32>) -> vec3<f32> {
    return grid_pos * uniforms.mesh_size / vec3<f32>(uniforms.field_size) + uniforms.chunk_offset;
}

// ============================================================================
// Pass 1: Generate Vertices
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn generate_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= FIELD_SIZE) {
        return;
    }

    let p = vec3<i32>(gid);

    // Sample 8 corners (using neighbor data for boundary)
    var corner_dists: array<f32, 8>;
    var num_negative = 0u;

    for (var i = 0u; i < 8u; i++) {
        let corner_offset = vec3<i32>(CUBE_CORNERS[i]);
        let sample_pos = p + corner_offset;
        corner_dists[i] = sample_sdf(sample_pos);
        if corner_dists[i] < 0.0 {
            num_negative++;
        }
    }

    let stride = linearize(gid);

    // No surface crossing
    if num_negative == 0u || num_negative == 8u {
        vertex_lookup[stride] = NULL_VERTEX;
        return;
    }

    // Calculate surface point
    let centroid = centroid_of_intersections(corner_dists);
    let grid_pos = vec3<f32>(gid) + centroid;
    let world_pos = grid_to_world(grid_pos);
    let normal = calculate_normal(corner_dists, centroid);

    // Allocate vertex
    let vertex_idx = atomicAdd(&counters.vertex_count, 1u);

    if vertex_idx < MAX_VERTICES {
        let base = vertex_idx * 3u;
        positions[base + 0u] = world_pos.x;
        positions[base + 1u] = world_pos.y;
        positions[base + 2u] = world_pos.z;

        normals[base + 0u] = normal.x;
        normals[base + 1u] = normal.y;
        normals[base + 2u] = normal.z;

        vertex_lookup[stride] = vertex_idx;
    }
}

// ============================================================================
// Pass 2: Generate Indices
// ============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, WORKGROUP_SIZE)
fn generate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Process all voxels including boundary (neighbors provide data)
    if any(gid >= FIELD_SIZE) {
        return;
    }

    let p = vec3<i32>(gid);
    let stride = linearize(gid);
    let v0 = vertex_lookup[stride];

    if v0 == NULL_VERTEX {
        return;
    }

    // Sample current and adjacent SDF values
    let d0 = sample_sdf(p);
    let dx = sample_sdf(p + vec3<i32>(1, 0, 0));
    let dy = sample_sdf(p + vec3<i32>(0, 1, 0));
    let dz = sample_sdf(p + vec3<i32>(0, 0, 1));

    let stride_x = 1u;
    let stride_y = FIELD_SIZE.x;
    let stride_z = FIELD_SIZE.x * FIELD_SIZE.y;

    // X-axis edge (connects voxels differing in Y and Z)
    if gid.y > 0u && gid.z > 0u && (d0 < 0.0) != (dx < 0.0) {
        let v1 = vertex_lookup[stride - stride_y];
        let v2 = vertex_lookup[stride - stride_z];
        let v3 = vertex_lookup[stride - stride_y - stride_z];

        if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
            emit_quad(v0, v1, v2, v3, d0 < 0.0);
        }
    }

    // Y-axis edge
    if gid.x > 0u && gid.z > 0u && (d0 < 0.0) != (dy < 0.0) {
        let v1 = vertex_lookup[stride - stride_z];
        let v2 = vertex_lookup[stride - stride_x];
        let v3 = vertex_lookup[stride - stride_x - stride_z];

        if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
            emit_quad(v0, v1, v2, v3, d0 < 0.0);
        }
    }

    // Z-axis edge
    if gid.x > 0u && gid.y > 0u && (d0 < 0.0) != (dz < 0.0) {
        let v1 = vertex_lookup[stride - stride_x];
        let v2 = vertex_lookup[stride - stride_y];
        let v3 = vertex_lookup[stride - stride_x - stride_y];

        if v1 != NULL_VERTEX && v2 != NULL_VERTEX && v3 != NULL_VERTEX {
            emit_quad(v0, v1, v2, v3, d0 < 0.0);
        }
    }
}

fn emit_quad(v0: u32, v1: u32, v2: u32, v3: u32, flip: bool) {
    let idx = atomicAdd(&counters.index_count, 6u);

    if idx + 6u <= MAX_INDICES {
        if flip {
            indices[idx + 0u] = v0;
            indices[idx + 1u] = v3;
            indices[idx + 2u] = v1;
            indices[idx + 3u] = v0;
            indices[idx + 4u] = v2;
            indices[idx + 5u] = v3;
        } else {
            indices[idx + 0u] = v0;
            indices[idx + 1u] = v1;
            indices[idx + 2u] = v3;
            indices[idx + 3u] = v0;
            indices[idx + 4u] = v3;
            indices[idx + 5u] = v2;
        }
    }
}
