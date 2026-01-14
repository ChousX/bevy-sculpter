# bevy-sculpter
[![Crates.io](https://img.shields.io/crates/v/bevy-sculpter.svg)](https://crates.io/crates/bevy-sculpter)
[![Docs.rs](https://docs.rs/bevy-sculpter/badge.svg)](https://docs.rs/bevy-sculpter)
[![License](https://img.shields.io/crates/l/bevy-sculpter.svg)](https://github.com/YOUR_USERNAME/bevy-sculpter#license)

![Screenshot](https://raw.githubusercontent.com/ChousX/bevy-sculpter/again/screenshots/screenshot0.png)

SDF-based voxel sculpting and Surface Nets meshing for Bevy.

> ⚠️ **Early Development**: This crate is under heavy development. Expect breaking changes between versions until 1.0.

## Features

- **Density Fields**: SDF-based volumetric representation with chunked storage
- **Surface Nets Meshing**: Smooth mesh generation from density fields with seamless chunk boundaries
- **Sculpting Brushes**: Smooth, hard (CSG), blur, and flatten brushes
- **Raycasting**: Sphere-tracing and DDA-based raycasting into density fields
- **SDF Redistancing**: Fast Sweeping Method for restoring proper signed distance fields

## Bevy Compatibility

| bevy-sculpter | Bevy |
|---------------|------|
| 0.18          | 0.18 |
| 0.1           | 0.17 |

## Quick Start

```rust
use bevy::prelude::*;
use bevy_sculpter::prelude::*;
use chunky_bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ChunkyPlugin::default())
        .add_plugins(SurfaceNetsPlugin)
        .insert_resource(DensityFieldMeshSize(vec3(10., 10., 10.)))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    // Create a chunk with a sphere
    let mut field = DensityField::new();
    bevy_sculpter::helpers::fill_centered_sphere(&mut field, 12.0);
    
    commands.spawn((
        Chunk,
        ChunkPos(ivec3(0, 0, 0)),
        field,
        DensityFieldDirty,
    ));

    // Camera and lighting...
}
```

## Sculpting

The crate provides several brush types for modifying density fields:

```rust
use bevy_sculpter::helpers::*;

// Hard CSG operations (instant)
brush_sphere(&mut field, center, radius, true);  // Add material
brush_sphere(&mut field, center, radius, false); // Remove material

// Smooth continuous brushes (for held input)
brush_smooth_timed(&mut field, center, radius, rate, delta_time, falloff);

// Surface smoothing
brush_blur(&mut field, center, radius, strength, falloff);

// Terrain flattening
brush_flatten(&mut field, center, radius, target_height, strength, falloff);
```

## Examples

Run the interactive sculpting example:

```bash
cargo run --example sculpt
```

Controls:
- **Middle click + drag**: Rotate camera
- **Right click (hold)**: Add material
- **Left click (hold)**: Remove material
- **Ctrl + click**: Hard brush (instant CSG)
- **B**: Toggle brush mode (Smooth/Blur)
- **Scroll**: Adjust brush size
- **[ / ]**: Adjust brush strength
- **WASD/Space/Shift**: Move camera

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Acknowledgments

Surface Nets implementation inspired by [fast-surface-nets-rs](https://github.com/bonsairobo/fast-surface-nets-rs) by bonsairobo.
