use crate::{DENSITY_FIELD_SIZE, FIELD_VOLUME};
use bevy::prelude::*;

/// The density field (SDF). Negative = inside, Positive = outside.
#[derive(Component, Clone, Deref, DerefMut, Debug)]
pub struct DensityField(pub Vec<f32>);

impl Default for DensityField {
    fn default() -> Self {
        Self(vec![1.0; FIELD_VOLUME]) // All outside
    }
}

impl DensityField {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn filled(value: f32) -> Self {
        Self(vec![value; FIELD_VOLUME])
    }

    #[inline]
    pub fn index(x: u32, y: u32, z: u32) -> usize {
        (x + y * DENSITY_FIELD_SIZE.x + z * DENSITY_FIELD_SIZE.x * DENSITY_FIELD_SIZE.y) as usize
    }

    #[inline]
    pub fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && y >= 0
            && z >= 0
            && (x as u32) < DENSITY_FIELD_SIZE.x
            && (y as u32) < DENSITY_FIELD_SIZE.y
            && (z as u32) < DENSITY_FIELD_SIZE.z
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: f32) {
        if x < DENSITY_FIELD_SIZE.x && y < DENSITY_FIELD_SIZE.y && z < DENSITY_FIELD_SIZE.z {
            self.0[Self::index(x, y, z)] = value;
        }
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        if x < DENSITY_FIELD_SIZE.x && y < DENSITY_FIELD_SIZE.y && z < DENSITY_FIELD_SIZE.z {
            self.0[Self::index(x, y, z)]
        } else {
            1.0 // Outside = exterior
        }
    }

    /// Get with signed coords (for neighbor sampling)
    #[inline]
    pub fn get_signed(&self, x: i32, y: i32, z: i32) -> Option<f32> {
        if Self::in_bounds(x, y, z) {
            Some(self.0[Self::index(x as u32, y as u32, z as u32)])
        } else {
            None
        }
    }
}
/// Result of finding the nearest interior point in a density field.
#[derive(Clone, Copy, Debug)]
pub struct NearestInteriorResult {
    /// Grid coordinates of the nearest interior voxel
    pub grid_pos: UVec3,
    /// World-space position of the nearest interior voxel center
    pub world_pos: Vec3,
    /// The density value at this point (negative = inside)
    pub density: f32,
    /// Squared distance from query point to this voxel (in grid space)
    pub distance_sq: f32,
}

/// Ray-AABB intersection, returns entry distance or None if no hit.
fn ray_aabb_entry(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let inv_dir = 1.0 / dir;
    let t1 = (min - origin) * inv_dir;
    let t2 = (max - origin) * inv_dir;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);
    let t_enter = t_min.max_element();
    let t_exit = t_max.min_element();

    if t_enter <= t_exit && t_exit >= 0.0 {
        Some(t_enter.max(0.0))
    } else {
        None
    }
}

/// Sort three values ascending
#[inline]
fn sort3(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    let (b, c) = if b <= c { (b, c) } else { (c, b) };
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    (a, b, c)
}

/// Solve the Eikonal equation |∇d| = 1 using sorted neighbor distances
#[inline]
fn solve_eikonal(a: f32, b: f32, c: f32, h: f32) -> f32 {
    // 1D solution: d = a + h
    let d = a + h;
    if d <= b {
        return d;
    }

    // 2D solution: (d-a)² + (d-b)² = h²
    let sum = a + b;
    let diff_sq = (a - b) * (a - b);
    let discriminant = 2.0 * h * h - diff_sq;

    if discriminant >= 0.0 {
        let d = (sum + discriminant.sqrt()) * 0.5;
        if d <= c {
            return d;
        }
    }

    // 3D solution: (d-a)² + (d-b)² + (d-c)² = h²
    let sum = a + b + c;
    let sq_sum = a * a + b * b + c * c;
    let discriminant = sum * sum - 3.0 * (sq_sum - h * h);

    if discriminant >= 0.0 {
        (sum + discriminant.sqrt()) / 3.0
    } else {
        a + h
    }
}
impl DensityField {
    // =========================================================================
    // Nearest Interior Queries
    // =========================================================================

    /// Find the nearest voxel with negative density (inside mesh) via brute force.
    ///
    /// O(n³) - checks every voxel. Use `nearest_interior_bounded` or
    /// `raycast_dda` for better performance when possible.
    pub fn nearest_interior(
        &self,
        world_pos: Vec3,
        mesh_size: Vec3,
    ) -> Option<NearestInteriorResult> {
        let scale = DENSITY_FIELD_SIZE.as_vec3() / mesh_size;
        let grid_pos = world_pos * scale;
        let inv_scale = mesh_size / DENSITY_FIELD_SIZE.as_vec3();

        let mut best: Option<NearestInteriorResult> = None;

        for z in 0..DENSITY_FIELD_SIZE.z {
            for y in 0..DENSITY_FIELD_SIZE.y {
                for x in 0..DENSITY_FIELD_SIZE.x {
                    let density = self.get(x, y, z);
                    if density >= 0.0 {
                        continue;
                    }

                    let voxel_center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    let dist_sq = grid_pos.distance_squared(voxel_center);

                    if best.as_ref().is_none_or(|b| dist_sq < b.distance_sq) {
                        best = Some(NearestInteriorResult {
                            grid_pos: UVec3::new(x, y, z),
                            world_pos: voxel_center * inv_scale,
                            density,
                            distance_sq: dist_sq,
                        });
                    }
                }
            }
        }

        best
    }

    /// Find nearest interior voxel within a maximum search radius.
    ///
    /// More efficient than `nearest_interior` for localized queries.
    pub fn nearest_interior_bounded(
        &self,
        world_pos: Vec3,
        mesh_size: Vec3,
        max_radius: f32,
    ) -> Option<NearestInteriorResult> {
        let scale = DENSITY_FIELD_SIZE.as_vec3() / mesh_size;
        let inv_scale = mesh_size / DENSITY_FIELD_SIZE.as_vec3();
        let grid_pos = world_pos * scale;
        let grid_radius = max_radius * scale.max_element();

        let min = (grid_pos - Vec3::splat(grid_radius))
            .max(Vec3::ZERO)
            .as_uvec3();
        let max = (grid_pos + Vec3::splat(grid_radius))
            .min(DENSITY_FIELD_SIZE.as_vec3() - Vec3::ONE)
            .as_uvec3();

        let max_dist_sq = grid_radius * grid_radius;
        let mut best: Option<NearestInteriorResult> = None;

        for z in min.z..=max.z {
            for y in min.y..=max.y {
                for x in min.x..=max.x {
                    let density = self.get(x, y, z);
                    if density >= 0.0 {
                        continue;
                    }

                    let voxel_center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    let dist_sq = grid_pos.distance_squared(voxel_center);

                    if dist_sq > max_dist_sq {
                        continue;
                    }

                    if best.as_ref().is_none_or(|b| dist_sq < b.distance_sq) {
                        best = Some(NearestInteriorResult {
                            grid_pos: UVec3::new(x, y, z),
                            world_pos: voxel_center * inv_scale,
                            density,
                            distance_sq: dist_sq,
                        });
                    }
                }
            }
        }

        best
    }

    /// Find the deepest interior voxel (most negative density).
    pub fn deepest_interior(&self, mesh_size: Vec3) -> Option<NearestInteriorResult> {
        let inv_scale = mesh_size / DENSITY_FIELD_SIZE.as_vec3();
        let mut best: Option<(UVec3, f32)> = None;

        for z in 0..DENSITY_FIELD_SIZE.z {
            for y in 0..DENSITY_FIELD_SIZE.y {
                for x in 0..DENSITY_FIELD_SIZE.x {
                    let density = self.get(x, y, z);
                    if density >= 0.0 {
                        continue;
                    }

                    if best.as_ref().is_none_or(|(_, d)| density < *d) {
                        best = Some((UVec3::new(x, y, z), density));
                    }
                }
            }
        }

        best.map(|(grid_pos, density)| {
            let voxel_center = grid_pos.as_vec3() + Vec3::splat(0.5);
            NearestInteriorResult {
                grid_pos,
                world_pos: voxel_center * inv_scale,
                density,
                distance_sq: 0.0,
            }
        })
    }

    // =========================================================================
    // Raycasting
    // =========================================================================

    /// Find interior point by sphere-tracing along a ray.
    ///
    /// Fast for well-formed SDFs. Uses density values as step sizes.
    /// For non-Euclidean fields, use `raycast_dda` instead.
    pub fn raycast_to_interior(
        &self,
        world_pos: Vec3,
        direction: Vec3,
        mesh_size: Vec3,
        max_steps: u32,
    ) -> Option<NearestInteriorResult> {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }

        let scale = DENSITY_FIELD_SIZE.as_vec3() / mesh_size;
        let inv_scale = mesh_size / DENSITY_FIELD_SIZE.as_vec3();
        let grid_dir = (dir * scale).normalize();

        let mut grid_pos = world_pos * scale;
        let bounds_max = DENSITY_FIELD_SIZE.as_vec3() - Vec3::ONE;

        // Advance to volume entry if starting outside
        if !Self::in_bounds(grid_pos.x as i32, grid_pos.y as i32, grid_pos.z as i32) {
            if let Some(t) = ray_aabb_entry(grid_pos, grid_dir, Vec3::ZERO, bounds_max) {
                grid_pos += grid_dir * (t + 0.01);
            } else {
                return None;
            }
        }

        let origin_grid = world_pos * scale;
        let min_step = 0.5;

        for _ in 0..max_steps {
            let ix = grid_pos.x as i32;
            let iy = grid_pos.y as i32;
            let iz = grid_pos.z as i32;

            if !Self::in_bounds(ix, iy, iz) {
                return None;
            }

            let density = self.get(ix as u32, iy as u32, iz as u32);

            if density < 0.0 {
                return Some(NearestInteriorResult {
                    grid_pos: UVec3::new(ix as u32, iy as u32, iz as u32),
                    world_pos: grid_pos * inv_scale,
                    density,
                    distance_sq: origin_grid.distance_squared(grid_pos),
                });
            }

            let step = density.max(min_step).min(8.0);
            grid_pos += grid_dir * step;
        }

        None
    }

    /// Sphere-trace toward field center.
    pub fn raycast_to_center(
        &self,
        world_pos: Vec3,
        mesh_size: Vec3,
        max_steps: u32,
    ) -> Option<NearestInteriorResult> {
        let center = mesh_size * 0.5;
        self.raycast_to_interior(world_pos, center - world_pos, mesh_size, max_steps)
    }

    /// Find interior point using DDA (3D Bresenham) traversal.
    ///
    /// Robust for any density field - doesn't rely on SDF values for stepping.
    /// Visits every voxel the ray passes through.
    pub fn raycast_dda(
        &self,
        world_pos: Vec3,
        direction: Vec3,
        mesh_size: Vec3,
        max_steps: u32,
    ) -> Option<NearestInteriorResult> {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }

        let scale = DENSITY_FIELD_SIZE.as_vec3() / mesh_size;
        let inv_scale = mesh_size / DENSITY_FIELD_SIZE.as_vec3();
        let grid_dir = (dir * scale).normalize();

        let mut grid_pos = world_pos * scale;
        let bounds_max = DENSITY_FIELD_SIZE.as_vec3();

        // Advance to volume entry if starting outside
        if !Self::in_bounds(grid_pos.x as i32, grid_pos.y as i32, grid_pos.z as i32) {
            if let Some(t) = ray_aabb_entry(grid_pos, grid_dir, Vec3::ZERO, bounds_max) {
                grid_pos += grid_dir * (t + 0.001);
            } else {
                return None;
            }
        }

        // DDA setup
        let mut voxel = grid_pos.floor().as_ivec3();
        let step = IVec3::new(
            if grid_dir.x >= 0.0 { 1 } else { -1 },
            if grid_dir.y >= 0.0 { 1 } else { -1 },
            if grid_dir.z >= 0.0 { 1 } else { -1 },
        );

        let next_boundary = Vec3::new(
            if grid_dir.x >= 0.0 {
                (voxel.x + 1) as f32
            } else {
                voxel.x as f32
            },
            if grid_dir.y >= 0.0 {
                (voxel.y + 1) as f32
            } else {
                voxel.y as f32
            },
            if grid_dir.z >= 0.0 {
                (voxel.z + 1) as f32
            } else {
                voxel.z as f32
            },
        );

        let mut t_max = Vec3::new(
            if grid_dir.x.abs() > 1e-10 {
                (next_boundary.x - grid_pos.x) / grid_dir.x
            } else {
                f32::MAX
            },
            if grid_dir.y.abs() > 1e-10 {
                (next_boundary.y - grid_pos.y) / grid_dir.y
            } else {
                f32::MAX
            },
            if grid_dir.z.abs() > 1e-10 {
                (next_boundary.z - grid_pos.z) / grid_dir.z
            } else {
                f32::MAX
            },
        );

        let t_delta = Vec3::new(
            if grid_dir.x.abs() > 1e-10 {
                (1.0 / grid_dir.x).abs()
            } else {
                f32::MAX
            },
            if grid_dir.y.abs() > 1e-10 {
                (1.0 / grid_dir.y).abs()
            } else {
                f32::MAX
            },
            if grid_dir.z.abs() > 1e-10 {
                (1.0 / grid_dir.z).abs()
            } else {
                f32::MAX
            },
        );

        let origin_grid = world_pos * scale;

        for _ in 0..max_steps {
            if Self::in_bounds(voxel.x, voxel.y, voxel.z) {
                let density = self.get(voxel.x as u32, voxel.y as u32, voxel.z as u32);

                if density < 0.0 {
                    let voxel_center = voxel.as_vec3() + Vec3::splat(0.5);
                    return Some(NearestInteriorResult {
                        grid_pos: voxel.as_uvec3(),
                        world_pos: voxel_center * inv_scale,
                        density,
                        distance_sq: origin_grid.distance_squared(voxel_center),
                    });
                }
            } else {
                return None;
            }

            // Step to next voxel
            if t_max.x < t_max.y && t_max.x < t_max.z {
                voxel.x += step.x;
                t_max.x += t_delta.x;
            } else if t_max.y < t_max.z {
                voxel.y += step.y;
                t_max.y += t_delta.y;
            } else {
                voxel.z += step.z;
                t_max.z += t_delta.z;
            }
        }

        None
    }

    /// DDA raycast toward field center.
    pub fn raycast_dda_to_center(
        &self,
        world_pos: Vec3,
        mesh_size: Vec3,
        max_steps: u32,
    ) -> Option<NearestInteriorResult> {
        let center = mesh_size * 0.5;
        self.raycast_dda(world_pos, center - world_pos, mesh_size, max_steps)
    }

    // =========================================================================
    // SDF Redistancing (Fast Sweeping Method)
    // =========================================================================

    /// Redistribute the SDF to restore proper Euclidean distances.
    ///
    /// Uses the Fast Sweeping Method to compute accurate signed distances
    /// from the current zero-crossing (surface). Call after CSG operations,
    /// blending, or manual edits that break the distance field property.
    ///
    /// # Performance
    /// - 32³ field: ~0.5-1ms
    /// - 64³ field: ~4-8ms
    pub fn redistribute_sdf(&mut self) {
        let sx = DENSITY_FIELD_SIZE.x as usize;
        let sy = DENSITY_FIELD_SIZE.y as usize;
        let sz = DENSITY_FIELD_SIZE.z as usize;

        // Step 1: Initialize - surface voxels get small values, others get ±infinity
        let mut dist = vec![f32::MAX; FIELD_VOLUME];

        for z in 0..sz {
            for y in 0..sy {
                for x in 0..sx {
                    let idx = x + y * sx + z * sx * sy;
                    let val = self.0[idx];
                    let sign = val < 0.0;

                    // Check 6-neighbors for sign change (surface detection)
                    let neighbors = [
                        (x > 0, x.wrapping_sub(1), y, z),
                        (x < sx - 1, x + 1, y, z),
                        (y > 0, x, y.wrapping_sub(1), z),
                        (y < sy - 1, x, y + 1, z),
                        (z > 0, x, y, z.wrapping_sub(1)),
                        (z < sz - 1, x, y, z + 1),
                    ];

                    let on_surface = neighbors.iter().any(|(valid, nx, ny, nz)| {
                        *valid && (self.0[nx + ny * sx + nz * sx * sy] < 0.0) != sign
                    });

                    dist[idx] = if on_surface {
                        val.abs().min(0.5)
                    } else if sign {
                        -f32::MAX
                    } else {
                        f32::MAX
                    };
                }
            }
        }

        // Step 2: Eight directional sweeps
        let h = 1.0f32;
        let sweeps: [(
            isize,
            isize,
            isize,
            isize,
            isize,
            isize,
            isize,
            isize,
            isize,
        ); 8] = [
            (0, sx as isize, 1, 0, sy as isize, 1, 0, sz as isize, 1),
            (
                sx as isize - 1,
                -1,
                -1,
                0,
                sy as isize,
                1,
                0,
                sz as isize,
                1,
            ),
            (
                0,
                sx as isize,
                1,
                sy as isize - 1,
                -1,
                -1,
                0,
                sz as isize,
                1,
            ),
            (
                0,
                sx as isize,
                1,
                0,
                sy as isize,
                1,
                sz as isize - 1,
                -1,
                -1,
            ),
            (
                sx as isize - 1,
                -1,
                -1,
                sy as isize - 1,
                -1,
                -1,
                0,
                sz as isize,
                1,
            ),
            (
                sx as isize - 1,
                -1,
                -1,
                0,
                sy as isize,
                1,
                sz as isize - 1,
                -1,
                -1,
            ),
            (
                0,
                sx as isize,
                1,
                sy as isize - 1,
                -1,
                -1,
                sz as isize - 1,
                -1,
                -1,
            ),
            (
                sx as isize - 1,
                -1,
                -1,
                sy as isize - 1,
                -1,
                -1,
                sz as isize - 1,
                -1,
                -1,
            ),
        ];

        for (x0, x1, dx, y0, y1, dy, z0, z1, dz) in sweeps {
            let mut z = z0;
            while z != z1 {
                let mut y = y0;
                while y != y1 {
                    let mut x = x0;
                    while x != x1 {
                        let xu = x as usize;
                        let yu = y as usize;
                        let zu = z as usize;
                        let idx = xu + yu * sx + zu * sx * sy;

                        let current = dist[idx];

                        // Get minimum neighbor distance along each axis
                        let a = [
                            if xu > 0 {
                                dist[idx - 1].abs()
                            } else {
                                f32::MAX
                            },
                            if xu < sx - 1 {
                                dist[idx + 1].abs()
                            } else {
                                f32::MAX
                            },
                        ]
                        .into_iter()
                        .fold(f32::MAX, f32::min);

                        let b = [
                            if yu > 0 {
                                dist[idx - sx].abs()
                            } else {
                                f32::MAX
                            },
                            if yu < sy - 1 {
                                dist[idx + sx].abs()
                            } else {
                                f32::MAX
                            },
                        ]
                        .into_iter()
                        .fold(f32::MAX, f32::min);

                        let c = [
                            if zu > 0 {
                                dist[idx - sx * sy].abs()
                            } else {
                                f32::MAX
                            },
                            if zu < sz - 1 {
                                dist[idx + sx * sy].abs()
                            } else {
                                f32::MAX
                            },
                        ]
                        .into_iter()
                        .fold(f32::MAX, f32::min);

                        let (a, b, c) = sort3(a, b, c);
                        let new_dist = solve_eikonal(a, b, c, h);

                        if new_dist < current.abs() {
                            dist[idx] = if current < 0.0 { -new_dist } else { new_dist };
                        }

                        x += dx;
                    }
                    y += dy;
                }
                z += dz;
            }
        }

        // Step 3: Write back preserving original signs
        for i in 0..FIELD_VOLUME {
            let sign = if self.0[i] < 0.0 { -1.0 } else { 1.0 };
            self.0[i] = sign * dist[i].abs();
        }
    }

    /// Check SDF quality by measuring gradient magnitude deviation from 1.0.
    ///
    /// Returns average error. Values close to 0.0 = proper distance field.
    /// Values > 0.3 suggest `redistribute_sdf()` may help.
    pub fn sdf_quality(&self) -> f32 {
        let (sx, sy, sz) = (
            DENSITY_FIELD_SIZE.x,
            DENSITY_FIELD_SIZE.y,
            DENSITY_FIELD_SIZE.z,
        );
        let mut total_error = 0.0f32;
        let mut count = 0u32;

        for z in 1..sz - 1 {
            for y in 1..sy - 1 {
                for x in 1..sx - 1 {
                    let dx = self.get(x + 1, y, z) - self.get(x - 1, y, z);
                    let dy = self.get(x, y + 1, z) - self.get(x, y - 1, z);
                    let dz = self.get(x, y, z + 1) - self.get(x, y, z - 1);

                    let grad_mag = (dx * dx + dy * dy + dz * dz).sqrt() * 0.5;
                    total_error += (grad_mag - 1.0).abs();
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_error / count as f32
        } else {
            0.0
        }
    }
}

/// Marker: this chunk needs remeshing
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct DensityFieldDirty;
