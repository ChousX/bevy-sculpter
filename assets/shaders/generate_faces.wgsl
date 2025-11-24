// ============================================
// KERNEL 4: Generate Faces
// ============================================
// This shader creates quad faces between adjacent vertices in the 3D grid.
// Following the Surface Nets algorithm, faces are only generated where
// the isosurface crosses between cells (sign change in SDF).

// STEP 1: Define bind group
@group(0) @binding(0)
var<storage, read> vertex_valid: array<u32>;  // Input: which cells have vertices

@group(0) @binding(1)
var<storage, read> vertex_indices: array<u32>;  // Input: compacted vertex indices

@group(0) @binding(2)
var<storage, read_write> faces: array<u32>;  // Output: face data (4 vertex indices per face)

@group(0) @binding(3)
var<storage, read_write> face_valid: array<u32>;  // Output: which face slots are valid

@group(0) @binding(4)
var<uniform> dimensions: vec3<u32>;  // Grid dimensions

// ===========================================================
// Helper function to convert 3D coordinates to 1D array index
// ===========================================================
fn get_cell_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * dimensions.x + z * dimensions.x * dimensions.y;
}

// STEP 2: Define workgroup size
// 8x8x8 = 512 threads per workgroup for 3D grid processing
@compute @workgroup_size(8, 8, 8)
fn generate_faces(
    @builtin(global_invocation_id) cell: vec3<u32>,
) {
    // STEP 3: Boundary check
    // Surface Nets doesn't generate faces on the maximum boundaries
    // This matches the Rust implementation's boundary handling
    if (cell.x >= dimensions.x - 1u || 
        cell.y >= dimensions.y - 1u || 
        cell.z >= dimensions.z - 1u) {
        return;
    }
    
    // STEP 4: Calculate this cell's index in the flattened array
    let cell_index = get_cell_index(cell.x, cell.y, cell.z);
    
    // STEP 5: Skip if this cell has no vertex
    // We can't create faces starting from a cell without a vertex
    if (vertex_valid[cell_index] == 0u) {
        return;
    }
    
    // STEP 6: Get the compacted vertex index for this cell
    // This is the actual index in the final vertex buffer
    let v0 = vertex_indices[cell_index];
    
    // STEP 7: Calculate base face index for this cell
    // Each cell reserves 3 face slots (one per axis: X, Y, Z)
    // Example: Cell 0 → faces 0,1,2; Cell 1 → faces 3,4,5; etc.
    let base_face_index = cell_index * 3u;
    var local_face_count = 0u;  // Tracks how many faces we actually create
    
    // ============================================
    // SURFACE NETS FACE GENERATION ALGORITHM
    // ============================================
    // Faces are generated along each axis direction (X, Y, Z).
    // For each axis, we create a quad face perpendicular to that axis
    // if all 4 cells forming that quad have valid vertices.
    //
    // The algorithm follows these principles from the Rust implementation:
    // 1. Only generate faces on edges parallel to each axis
    // 2. Check boundaries: don't generate on minimum Y/Z (for X-faces), etc.
    // 3. Verify all 4 corner cells have vertices before creating the quad
    //
    // Quad vertex ordering (counter-clockwise when viewed from outside):
    //    v3 ---- v2
    //    |       |
    //    v0 ---- v1
    
    // ============================================
    // FACE 1: X-axis parallel edge (Y-Z plane face)
    // ============================================
    // Generate a quad perpendicular to the X-axis
    // This face lies in a Y-Z plane and requires 4 cells:
    //   v0: current cell      (x, y,   z)
    //   v1: +Y neighbor       (x, y+1, z)
    //   v2: +Y+Z neighbor     (x, y+1, z+1)
    //   v3: +Z neighbor       (x, y,   z+1)
    //
    // Boundary check: Only create if not on minimum Y or Z boundaries
    // (matches: if x != minx && y != miny in Rust)
    if (cell.y > 0u && cell.z > 0u) {
        // Calculate indices of the 3 neighboring cells needed for this quad
        let idx1 = get_cell_index(cell.x, cell.y + 1u, cell.z);       // +Y neighbor
        let idx2 = get_cell_index(cell.x, cell.y + 1u, cell.z + 1u);  // +Y+Z neighbor
        let idx3 = get_cell_index(cell.x, cell.y,       cell.z + 1u);  // +Z neighbor
        
        // Check if all 4 cells have valid vertices
        if (vertex_valid[idx1] != 0u && 
            vertex_valid[idx2] != 0u && 
            vertex_valid[idx3] != 0u) {
            
            // Get compacted vertex indices for all 4 corners
            let v1 = vertex_indices[idx1];
            let v2 = vertex_indices[idx2];
            let v3 = vertex_indices[idx3];
            
            // Calculate where to write this face in the output array
            let face_idx = base_face_index + local_face_count;
            let face_data_base = face_idx * 4u;  // Each face stores 4 vertex indices
            
            // Write the quad vertices in counter-clockwise order
            faces[face_data_base + 0u] = v0;  // Bottom-left
            faces[face_data_base + 1u] = v1;  // Top-left (moved up in Y)
            faces[face_data_base + 2u] = v2;  // Top-right (moved up in Y and Z)
            faces[face_data_base + 3u] = v3;  // Bottom-right (moved up in Z)
            
            // Mark this face slot as containing valid data
            face_valid[face_idx] = 1u;
            local_face_count = local_face_count + 1u;
        }
    }
    
    // ============================================
    // FACE 2: Y-axis parallel edge (X-Z plane face)
    // ============================================
    // Generate a quad perpendicular to the Y-axis
    // This face lies in an X-Z plane and requires 4 cells:
    //   v0: current cell      (x,   y, z)
    //   v1: +X neighbor       (x+1, y, z)
    //   v2: +X+Z neighbor     (x+1, y, z+1)
    //   v3: +Z neighbor       (x,   y, z+1)
    //
    // Boundary check: Only create if not on minimum X or Z boundaries
    // (matches: if x != minx && z != minz in Rust)
    if (cell.x > 0u && cell.z > 0u) {
        // Calculate indices of the 3 neighboring cells
        let idx1 = get_cell_index(cell.x + 1u, cell.y, cell.z);       // +X neighbor
        let idx2 = get_cell_index(cell.x + 1u, cell.y, cell.z + 1u);  // +X+Z neighbor
        let idx3 = get_cell_index(cell.x,       cell.y, cell.z + 1u);  // +Z neighbor
        
        // Verify all 4 corner vertices exist
        if (vertex_valid[idx1] != 0u && 
            vertex_valid[idx2] != 0u && 
            vertex_valid[idx3] != 0u) {
            
            // Get vertex indices
            let v1 = vertex_indices[idx1];
            let v2 = vertex_indices[idx2];
            let v3 = vertex_indices[idx3];
            
            // Write face data
            let face_idx = base_face_index + local_face_count;
            let face_data_base = face_idx * 4u;
            
            faces[face_data_base + 0u] = v0;  // Bottom-left
            faces[face_data_base + 1u] = v1;  // Bottom-right (moved in X)
            faces[face_data_base + 2u] = v2;  // Top-right (moved in X and Z)
            faces[face_data_base + 3u] = v3;  // Top-left (moved in Z)
            
            // Mark face as valid
            face_valid[face_idx] = 1u;
            local_face_count = local_face_count + 1u;
        }
    }
    
    // ============================================
    // FACE 3: Z-axis parallel edge (X-Y plane face)
    // ============================================
    // Generate a quad perpendicular to the Z-axis
    // This face lies in an X-Y plane and requires 4 cells:
    //   v0: current cell      (x,   y,   z)
    //   v1: +X neighbor       (x+1, y,   z)
    //   v2: +X+Y neighbor     (x+1, y+1, z)
    //   v3: +Y neighbor       (x,   y+1, z)
    //
    // Boundary check: Only create if not on minimum X or Y boundaries
    // (matches: if y != miny && z != minz in Rust)
    if (cell.x > 0u && cell.y > 0u) {
        // Calculate indices of the 3 neighboring cells
        let idx1 = get_cell_index(cell.x + 1u, cell.y,       cell.z);  // +X neighbor
        let idx2 = get_cell_index(cell.x + 1u, cell.y + 1u, cell.z);  // +X+Y neighbor
        let idx3 = get_cell_index(cell.x,       cell.y + 1u, cell.z);  // +Y neighbor
        
        // Verify all 4 corner vertices exist
        if (vertex_valid[idx1] != 0u && 
            vertex_valid[idx2] != 0u && 
            vertex_valid[idx3] != 0u) {
            
            // Get vertex indices
            let v1 = vertex_indices[idx1];
            let v2 = vertex_indices[idx2];
            let v3 = vertex_indices[idx3];
            
            // Write face data
            let face_idx = base_face_index + local_face_count;
            let face_data_base = face_idx * 4u;
            
            faces[face_data_base + 0u] = v0;  // Bottom-left
            faces[face_data_base + 1u] = v1;  // Bottom-right (moved in X)
            faces[face_data_base + 2u] = v2;  // Top-right (moved in X and Y)
            faces[face_data_base + 3u] = v3;  // Top-left (moved in Y)
            
            // Mark face as valid
            face_valid[face_idx] = 1u;
            local_face_count = local_face_count + 1u;
        }
    }
    
    // STEP 8: Mark unused face slots as invalid
    // Each cell reserves 3 face slots, but may use fewer
    // Mark any unused slots (when local_face_count < 3) as invalid
    // This ensures the compaction step knows which faces to keep
    for (var i = local_face_count; i < 3u; i = i + 1u) {
        face_valid[base_face_index + i] = 0u;
    }
}

// ============================================
// ALGORITHM SUMMARY
// ============================================
// This shader implements the face generation step of Surface Nets:
//
// 1. For each cell with a valid vertex (surface intersection)
// 2. Try to create up to 3 quad faces (one per axis direction)
// 3. Each face requires 4 adjacent cells to all have vertices
// 4. Faces are only created on interior edges (not on grid boundaries)
// 5. The result is a set of quads that form the mesh surface
//
// The quads will be converted to triangles in a later step:
// Each quad [v0, v1, v2, v3] becomes two triangles:
//   Triangle 1: [v0, v1, v2]
//   Triangle 2: [v0, v2, v3]
