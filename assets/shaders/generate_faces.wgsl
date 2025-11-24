// ============================================
// KERNEL 4: Generate Faces
// ============================================
// This shader creates quad faces between adjacent vertices in the 3D grid.
// Each cell can generate up to 3 faces (one for each axis direction).

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
// Helper function MUST be at global scope in WGSL
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
    // STEP 4: Boundary check
    // We need to access neighboring cells, so we need to be within bounds
    if (cell.x >= dimensions.x - 1u || 
        cell.y >= dimensions.y - 1u || 
        cell.z >= dimensions.z - 1u) {
        return;
    }
    
    // STEP 5: Calculate cell index
    let cell_index = get_cell_index(cell.x, cell.y, cell.z);   

    // STEP 6: Skip if this cell has no vertex
    // Can't make faces if there's no vertex here
    if (vertex_valid[cell_index] == 0u) {
        return;
    }
    
    // STEP 7: Get the compacted vertex index for this cell
    let v0 = vertex_indices[cell_index];
   
    
    // STEP 9: Calculate base face index for this cell
    // Each cell can generate up to 3 faces, so we reserve 3 slots per cell
    // Cell 0 gets face slots 0,1,2; Cell 1 gets slots 3,4,5; etc.
    let base_face_index = cell_index * 3u;
    var local_face_count = 0u;  // Track how many faces we actually create
    
    // ============================================
    // FACE GENERATION EXPLANATION
    // ============================================
    // Surface Nets creates faces by connecting vertices in adjacent cells.
    // Each cell can create 3 quad faces in the positive X, Y, and Z directions.
    //
    // Quad layout for each face:
    //    v3 ---- v2
    //    |       |
    //    |       |
    //    v0 ---- v1
    //
    // We create a face if all 4 neighboring cells have valid vertices.
    
    // ============================================
    // FACE 1: X-Y Plane (looking in +Z direction)
    // ============================================
    // STEP 10: Create face in the X-Y plane
    // This face connects 4 cells in a square on the X-Y plane:
    //   v0: current cell (x,   y,   z)
    //   v1: right        (x+1, y,   z)
    //   v2: right-back   (x+1, y+1, z)
    //   v3: back         (x,   y+1, z)
    
    if (cell_x + 1u < dimensions.x - 1u && cell_y + 1u < dimensions.y - 1u) {
        // Calculate indices of the 3 neighboring cells
        let idx1 = get_cell_index(cell_x + 1u, cell_y,       cell_z);  // Right
        let idx2 = get_cell_index(cell_x + 1u, cell_y + 1u, cell_z);  // Right-back
        let idx3 = get_cell_index(cell_x,       cell_y + 1u, cell_z);  // Back
        
        // STEP 11: Check if all 4 cells have valid vertices
        if (vertex_valid[idx1] != 0u && 
            vertex_valid[idx2] != 0u && 
            vertex_valid[idx3] != 0u) {
            
            // Get compacted vertex indices for all 4 corners
            let v1 = vertex_indices[idx1];
            let v2 = vertex_indices[idx2];
            let v3 = vertex_indices[idx3];
            
            // STEP 12: Write face to output
            // Each face is stored as 4 consecutive u32 values (vertex indices)
            let face_idx = base_face_index + local_face_count;
            let face_data_base = face_idx * 4u;
            
            faces[face_data_base + 0u] = v0;  // Bottom-left
            faces[face_data_base + 1u] = v1;  // Bottom-right
            faces[face_data_base + 2u] = v2;  // Top-right
            faces[face_data_base + 3u] = v3;  // Top-left
            
            // Mark this face slot as valid
            face_valid[face_idx] = 1u;
            local_face_count = local_face_count + 1u;
        }
    }
    
    // ============================================
    // FACE 2: X-Z Plane (looking in +Y direction)
    // ============================================
    // STEP 13: Create face in the X-Z plane
    // This face connects 4 cells in a square on the X-Z plane:
    //   v0: current cell (x,   y, z)
    //   v1: right        (x+1, y, z)
    //   v2: right-top    (x+1, y, z+1)
    //   v3: top          (x,   y, z+1)
    
    if (cell.x + 1u < dimensions.x - 1u && cell.z + 1u < dimensions.z - 1u) {
        let idx1 = get_cell_index(cell.x + 1u, cell.y, cell.z);        // Right
        let idx2 = get_cell_index(cell.x + 1u, cell.y, cell.z + 1u);  // Right-top
        let idx3 = get_cell_index(cell.x,       cell.y, cell.z + 1u);  // Top
        
        if (vertex_valid[idx1] != 0u && 
            vertex_valid[idx2] != 0u && 
            vertex_valid[idx3] != 0u) {
            
            let v1 = vertex_indices[idx1];
            let v2 = vertex_indices[idx2];
            let v3 = vertex_indices[idx3];
            
            let face_idx = base_face_index + local_face_count;
            let face_data_base = face_idx * 4u;
            
            faces[face_data_base + 0u] = v0;
            faces[face_data_base + 1u] = v1;
            faces[face_data_base + 2u] = v2;
            faces[face_data_base + 3u] = v3;
            
            face_valid[face_idx] = 1u;
            local_face_count = local_face_count + 1u;
        }
    }
    
    // ============================================
    // FACE 3: Y-Z Plane (looking in +X direction)
    // ============================================
    // STEP 14: Create face in the Y-Z plane
    // This face connects 4 cells in a square on the Y-Z plane:
    //   v0: current cell (x, y,   z)
    //   v1: back         (x, y+1, z)
    //   v2: back-top     (x, y+1, z+1)
    //   v3: top          (x, y,   z+1)
    
    if (cell.y + 1u < dimensions.y - 1u && cell.z + 1u < dimensions.z - 1u) {
        let idx1 = get_cell_index(cell.x, cell.y + 1u, cell.z);        // Back
        let idx2 = get_cell_index(cell.x, cell.y + 1u, cell.z + 1u);  // Back-top
        let idx3 = get_cell_index(cell.x, cell.y,       cell.z + 1u);  // Top
        
        if (vertex_valid[idx1] != 0u && 
            vertex_valid[idx2] != 0u && 
            vertex_valid[idx3] != 0u) {
            
            let v1 = vertex_indices[idx1];
            let v2 = vertex_indices[idx2];
            let v3 = vertex_indices[idx3];
            
            let face_idx = base_face_index + local_face_count;
            let face_data_base = face_idx * 4u;
            
            faces[face_data_base + 0u] = v0;
            faces[face_data_base + 1u] = v1;
            faces[face_data_base + 2u] = v2;
            faces[face_data_base + 3u] = v3;
            
            face_valid[face_idx] = 1u;
            local_face_count = local_face_count + 1u;
        }
    }
    
    // STEP 15: Mark unused face slots as invalid
    // If we created fewer than 3 faces, mark the remaining slots as invalid
    for (var i = local_face_count; i < 3u; i = i + 1u) {
        face_valid[base_face_index + i] = 0u;
    }
}

// ============================================
// EXAMPLE VISUALIZATION
// ============================================
// Imagine a 3x3x3 grid of cells. Each cell with a valid vertex can create
// faces with its neighbors. A cell at (1,1,1) might create:
//
// 1. X-Y face: connects (1,1,1), (2,1,1), (2,2,1), (1,2,1)
// 2. X-Z face: connects (1,1,1), (2,1,1), (2,1,2), (1,1,2)
// 3. Y-Z face: connects (1,1,1), (1,2,1), (1,2,2), (1,1,2)
//
// Each face is a quad (4 vertices) that represents part of the surface.
// The faces connect together to form the complete mesh surface.
