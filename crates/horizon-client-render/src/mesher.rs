//! Section meshing on worker threads. Faces follow the baked blockstate models
//! with vanilla's neighbour culling; vertex light and colour follow
//! `ModelBlockRenderer`/`BlockModelLighter` (smooth light, ambient occlusion,
//! cardinal shading, tint sources, block offsets; see `lighting`). Fluids
//! follow `FluidRenderer` (see `fluid`).
mod fluid;
mod lighting;
mod sides;
use crate::{
    assets::{Assets, RenderAssets},
    biome::Biomes,
    blocks::{BlockFacts, RenderShape},
    chunk::{Chunk, MIN_Y},
    models::catalog::BakedFace,
};
use horizon_core::{
    biome::BiomeZoom, geometry::Direction, position::BlockPos, random::position_seed,
};
use horizon_world::{
    block_render::FaceVisibility,
    fluid::{FluidKind, FluidState},
    registry::BlockStateId,
};
pub use sides::Sides;
use std::{collections::BTreeMap, sync::Arc};

/// An entity, item and effect vertex (the entity pipelines), with a face
/// normal for `entity.wgsl`'s directional light.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub layer: u32,
    /// RGBA8: tint multiplied by directional shade and ambient occlusion.
    pub color: u32,
    /// Smooth light, 16x fixed point: sky in bits 0..8, block in bits 8..16.
    pub light: u32,
    /// RGBA8_SNORM face normal; w = -127 for two-sided lighting, 0 to keep baked/unlit colour,
    /// [`CULL_BACK`] to keep it and also drop back-facing pixels.
    pub normal: [i8; 4],
    /// The `GLINT_TEXTURING` atlas UV for a foil item quad; zero elsewhere.
    pub glint_uv: [f32; 2],
}
/// `Vertex::normal` of a baked-colour quad of an `entitySolid` model: the render type culls
/// back faces, which the entity pipelines here do not, so `entity.wgsl` discards them.
pub const CULL_BACK: [i8; 4] = [0, 0, 0, 1];
/// A section vertex: vanilla's `DefaultVertexFormat.BLOCK` (position, colour,
/// UV0, UV2) with the sprite's array layer beside the light, in 24 bytes
/// rather than vanilla's 28: the UV is fixed point, which holds model UVs
/// (multiples of 1/16 and their halves) exactly. The position stays a float:
/// fixed point would move the edges of offset plants and rotated elements by
/// a fraction of a pixel. Terrain has no normal: `ModelBlockRenderer` bakes
/// the face shade into the colour and `terrain.vsh` only multiplies it by the
/// lightmap.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainVertex {
    /// Section-relative position.
    pub pos: [f32; 3],
    /// UV within the sprite, `UV_STEPS` per unit (`TerrainVertex::new`,
    /// `terrain.wgsl`'s `terrain_uv`).
    pub(crate) uv: [u16; 2],
    /// Sprite array layer, with `OPAQUE_LAYER` for opaque leaves and quads over opaque texels.
    pub layer: u16,
    /// Smooth light, 16x fixed point: sky in the low byte, block in the high one.
    pub light: u16,
    /// RGBA8: tint multiplied by directional shade and ambient occlusion.
    pub color: u32,
}
/// Fixed-point steps per unit of `TerrainVertex` UVs, which span `[0, 2)`.
pub const UV_STEPS: f32 = 32768.0;
/// Fixed-point steps per block of distant-terrain UVs (`TerrainVertex::tiled`),
/// which span `[0, 2048)`: merged faces repeat their sprite once per block.
pub const TILED_UV_STEPS: f32 = 32.0;
/// `TerrainVertex::layer` bit selecting block-scaled UVs for close-up greedy
/// quads. Sprite array indices are limited to the lower 14 bits by the
/// renderer's eight 2048-layer arrays; bit 15 remains `OPAQUE_LAYER`.
pub const TILED_UV_LAYER: u16 = 1 << 14;

impl TerrainVertex {
    pub fn new(pos: [f32; 3], uv: [f32; 2], layer: u16, color: u32, light: u16) -> Self {
        Self {
            pos,
            uv: uv.map(|v| (v * UV_STEPS).round().clamp(0.0, f32::from(u16::MAX)) as u16),
            layer,
            light,
            color,
        }
    }
    /// A vertex of a merged distant-terrain face, its UV in blocks from the
    /// face's corner. Drawn with an instance UV scale of `1 / TILED_UV_STEPS`.
    pub fn tiled(pos: [f32; 3], uv: [f32; 2], layer: u16, color: u32, light: u16) -> Self {
        Self {
            pos,
            uv: uv.map(|v| (v * TILED_UV_STEPS).round().clamp(0.0, f32::from(u16::MAX)) as u16),
            layer,
            light,
            color,
        }
    }
    /// A close-up greedy face. The vertex shader recognizes `TILED_UV_LAYER`
    /// and decodes this UV with the block-scaled step without changing the
    /// instance format used by ordinary model quads.
    pub fn tiled_section(
        pos: [f32; 3],
        uv: [f32; 2],
        layer: u16,
        color: u32,
        light: u16,
    ) -> Self {
        let mut vertex = Self::tiled(pos, uv, layer, color, light);
        vertex.layer |= TILED_UV_LAYER;
        vertex
    }

    /// The UV the GPU samples.
    pub fn uv(&self) -> [f32; 2] {
        self.uv.map(|v| f32::from(v) / UV_STEPS)
    }
}
#[cfg(test)]
mod vertex_tests {
    use super::TerrainVertex;
    #[test]
    fn model_uvs_pack_exactly() {
        for i in 0..=32 {
            let u = i as f32 / 32.0;
            let v = TerrainVertex::new([0.0; 3], [u, 1.0 - u], 0, 0, 0);
            assert_eq!(v.uv(), [u, 1.0 - u]);
        }
    }
    #[test]
    fn other_uvs_round_within_a_step() {
        // A flowing fluid's rotated UV.
        let uv = [0.5 + 0.25 * 0.3f32.sin(), 0.5 - 0.25 * 0.3f32.cos()];
        let packed = TerrainVertex::new([0.0; 3], uv, 0, 0, 0).uv();
        for (a, b) in packed.iter().zip(uv) {
            assert!((a - b).abs() <= 0.5 / 32768.0, "{a} {b}");
        }
    }
    #[test]
    fn emission_keeps_the_colour_and_the_free_bits() {
        let rgb = super::pack_rgb([0x12, 0x34, 0x56]);
        assert_eq!(super::emitting(rgb, 0), rgb);
        assert_eq!(super::emitting(rgb, 9), 0xF656_3412);
        assert_eq!(super::emitting(rgb, 15), 0xF056_3412);
        assert_eq!(super::emitting(rgb, 15) | super::EMISSION_BITS, rgb);
    }
}
pub struct SectionMesh {
    pub solid: Vec<TerrainVertex>,
    /// The buckets `solid` is sorted in by the side its quads face: empty
    /// until `sort_solid`, which the meshing workers call; `solid` is in
    /// model order before.
    pub sides: Sides,
    pub translucent: Vec<TerrainVertex>,
    /// VisGraph: bit `a * 6 + b` is set when faces `a` and `b` (Direction
    /// order) are connected through non-opaque blocks.
    pub visibility: u64,
    /// Blocks with a block-entity renderer (world position, state id), drawn
    /// each frame by `block_entity_render` like vanilla's per-section
    /// `SectionCompiler.Results.blockEntities`.
    pub block_entities: Vec<([i32; 3], u32)>,
}
impl Default for SectionMesh {
    fn default() -> Self {
        Self {
            solid: Vec::new(),
            sides: Sides::default(),
            translucent: Vec::new(),
            visibility: ALL_VISIBLE,
            block_entities: Vec::new(),
        }
    }
}
impl SectionMesh {
    /// Sorts the solid quads by the side they face, for the renderer to
    /// leave out those facing away from the eye (`sides`).
    pub fn sort_solid(&mut self) {
        self.sides = sides::sort(&mut self.solid);
    }
}
pub const ALL_VISIBLE: u64 = (1 << 36) - 1;
/// VisGraph.resolve: flood fill non-opaque cells and connect the faces each region touches.
fn visibility(pad: &Padded, info: &[Info]) -> u64 {
    // Opaque cells are already visited: resolve each state's opacity once,
    // then reuse this map for the flood fill and the uniform-section shortcuts.
    let mut seen = [false; 4096];
    let mut opaque_count = 0;
    for y in 0..16 {
        for z in 0..16 {
            let row = pi(0, y, z);
            for x in 0..16 {
                let opaque = info[pad.blocks[row + x] as usize].solid;
                seen[(y as usize) << 8 | (z as usize) << 4 | x] = opaque;
                opaque_count += usize::from(opaque);
            }
        }
    }
    if opaque_count == 0 {
        return ALL_VISIBLE;
    }
    if opaque_count == 4096 {
        return 0;
    }
    let mut result = 0u64;
    let mut stack = Vec::new();
    for start in 0..4096usize {
        if seen[start] {
            continue;
        }
        let mut faces = 0u8;
        seen[start] = true;
        stack.push(start);
        while let Some(c) = stack.pop() {
            let (x, y, z) = ((c & 15) as i32, (c >> 8) as i32, ((c >> 4) & 15) as i32);
            if y == 0 {
                faces |= 1 << 0;
            }
            if y == 15 {
                faces |= 1 << 1;
            }
            if z == 0 {
                faces |= 1 << 2;
            }
            if z == 15 {
                faces |= 1 << 3;
            }
            if x == 0 {
                faces |= 1 << 4;
            }
            if x == 15 {
                faces |= 1 << 5;
            }
            // A single connected region touching all six faces sets every
            // pair (including the diagonal); unvisited cavities cannot add more.
            if faces == 0b11_1111 {
                return ALL_VISIBLE;
            }
            for (dx, dy, dz) in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ] {
                let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                if !(0..16).contains(&nx) || !(0..16).contains(&ny) || !(0..16).contains(&nz) {
                    continue;
                }
                let n = (ny << 8 | nz << 4 | nx) as usize;
                if !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        for a in 0..6 {
            if faces & (1 << a) != 0 {
                for b in 0..6 {
                    if faces & (1 << b) != 0 {
                        result |= 1 << (a * 6 + b);
                    }
                }
            }
        }
        if result == ALL_VISIBLE {
            return result;
        }
    }
    result
}
#[cfg(test)]
mod particle_tests;
#[cfg(test)]
mod visibility_tests;
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tint {
    None,
    Grass,
    /// `BlockTintSources.grassBlock`: grass in the world, none on its particles.
    GrassBlock,
    TallGrassUpper,
    Foliage,
    DryFoliage,
    Water,
    Constant(u32),
    Redstone(u8),
    Stem(u8),
}
/// Per-state data the mesher needs on every block, flattened for speed.
#[derive(Clone, Copy)]
struct Info {
    /// `isSolidRender`.
    solid: bool,
    /// `isCollisionShapeFullBlock`: AO faces flush with a full block sample outside it.
    full: bool,
    /// `isLightPermeable`: diagonal AO samples are taken through such side blocks.
    permeable: bool,
    /// `getShadeBrightness` (ambient occlusion contribution).
    shade: f32,
    emission: u8,
    /// `emissiveRendering`: drawn and sampled at full brightness.
    emissive: bool,
    fluid: FluidState,
    model: bool,
    /// The state has a block entity (candidates for block-entity renderers).
    block_entity: bool,
    tints: [Tint; 2],
    tint_count: u8,
    /// `OffsetType` as (max horizontal, max vertical): vertical 0 for XZ.
    offset: Option<(f32, f32)>,
    /// `isSolid()` (legacy), for fluid corner heights.
    legacy_solid: bool,
    /// `HalfTransparentBlock`/`LeavesBlock`: water sides show the overlay sprite.
    overlay: bool,
    /// `LeavesBlock`: with opaque leaves the faces between two leaf blocks are skipped.
    leaves: bool,
    blocks_flow: bool,
    ice: bool,
    sturdy: u8,
    /// Occlusion face per direction: 0 empty, 1 the full-block singleton, 2 partial.
    face: [u8; 6],
    /// Height from the bottom over which face `d`, flush with the `d` side, covers
    /// the full width (vertical faces: 1 when the whole square is covered).
    cover: [f32; 6],
}
/// Derived per-state facts, exposed for the section-mesh parity gate.
#[derive(Clone, Copy, Debug)]
pub struct MeshFacts {
    pub shade: f32,
    pub emissive: bool,
    pub legacy_solid: bool,
    pub light_permeable: bool,
    pub solid_render: bool,
    pub full_collision: bool,
    pub emission: u8,
    pub overlay: bool,
    pub tint_sources: u8,
}
pub struct MeshTables {
    info: Vec<Info>,
    pub visibility: FaceVisibility,
    water_still: u16,
    water_flow: u16,
    water_overlay: u16,
    lava_still: u16,
    lava_flow: u16,
}
fn redstone_color(power: u8) -> u32 {
    // RedstoneWireBlock.COLORS
    let f = f32::from(power) / 15.0;
    let r = f * 0.6 + if f > 0.0 { 0.4 } else { 0.3 };
    let g = (f * f * 0.7 - 0.5).clamp(0.0, 1.0);
    let b = (f * f * 0.6 - 0.7).clamp(0.0, 1.0);
    ((r * 255.0) as u32) << 16 | ((g * 255.0) as u32) << 8 | (b * 255.0) as u32
}
impl MeshTables {
    pub fn new(
        assets: &Assets,
        render: &RenderAssets,
        facts: &BlockFacts,
        visibility: FaceVisibility,
    ) -> crate::Result<Self> {
        // Bit 14 selects tiled UVs and bit 15 marks opaque texels, so real
        // sprite layers must fit in the lower 14 bits.
        let layers = render.sprite_layers.values().max().map_or(0, |l| l + 1);
        if layers > u32::from(TILED_UV_LAYER) {
            return Err(format!(
                "{layers} block sprites, terrain vertices reserve layer bits above {}",
                TILED_UV_LAYER - 1
            )
            .into());
        }
        let layer = |name: &str| -> crate::Result<u16> {
            render
                .sprite_layers
                .get(name)
                .map(|&l| l as u16)
                .ok_or_else(|| format!("missing fluid sprite {name}").into())
        };
        let mut covers = std::collections::HashMap::new();
        let mut info = Vec::with_capacity(facts.states.len());
        for (id, s) in facts.states.iter().enumerate() {
            let state = assets
                .registry
                .state(BlockStateId(id as u32))
                .ok_or("missing registry state")?;
            let props = &state.properties;
            let prop = |k: &str| -> u8 { props.get(k).and_then(|v| v.parse().ok()).unwrap_or(0) };
            let name = s.name.strip_prefix("minecraft:").unwrap_or(&s.name);
            // BlockColors.createDefault: tint sources by block.
            let (tints, tint_count) = match name {
                "large_fern" | "tall_grass" => {
                    if props.get("half").map(String::as_str) == Some("upper") {
                        ([Tint::TallGrassUpper; 2], 1)
                    } else {
                        ([Tint::Grass; 2], 1)
                    }
                }
                "fern" | "short_grass" | "potted_fern" | "bush" | "sugar_cane" => {
                    ([Tint::Grass; 2], 1)
                }
                "grass_block" => ([Tint::GrassBlock; 2], 1),
                "pink_petals" | "wildflowers" => ([Tint::None, Tint::Grass], 2),
                "spruce_leaves" => ([Tint::Constant(0x61_9961); 2], 1),
                "birch_leaves" => ([Tint::Constant(0x80_A755); 2], 1),
                "oak_leaves" | "jungle_leaves" | "acacia_leaves" | "dark_oak_leaves" | "vine"
                | "mangrove_leaves" => ([Tint::Foliage; 2], 1),
                "leaf_litter" => ([Tint::DryFoliage; 2], 1),
                "water_cauldron" => ([Tint::Water; 2], 1),
                // Their in-world colour is the particle-only source's -1.
                "water" | "bubble_column" => ([Tint::None; 2], 1),
                "redstone_wire" => ([Tint::Redstone(prop("power")); 2], 1),
                "attached_melon_stem" | "attached_pumpkin_stem" => {
                    ([Tint::Constant(0xE0_C71C); 2], 1)
                }
                "melon_stem" | "pumpkin_stem" => ([Tint::Stem(prop("age")); 2], 1),
                "lily_pad" => ([Tint::Constant(0x20_8030); 2], 1),
                _ => ([Tint::None; 2], 0),
            };
            // getShadeBrightness overrides; the default is 0.2 for full collision shapes.
            let overrides = [
                "TransparentBlock",
                "BarrierBlock",
                "LightBlock",
                "StructureVoidBlock",
            ];
            let shade = if overrides.iter().any(|c| s.has_class(c)) {
                1.0
            } else if s.has_class("SoulSandBlock") || s.has_class("MudBlock") {
                0.2
            } else if s.has_class("SnowLayerBlock") {
                if prop("layers") == 8 { 0.2 } else { 1.0 }
            } else if s.full_collision {
                0.2
            } else {
                1.0
            };
            let emissive = name == "magma_block"
                || (matches!(name, "sculk_sensor" | "calibrated_sculk_sensor")
                    && props.get("sculk_sensor_phase").map(String::as_str) == Some("active"));
            // OffsetType.XZ/XYZ with getMaxHorizontalOffset/getMaxVerticalOffset.
            let offset = s.random_offset.then(|| {
                let horizontal = if s.has_class("SpeleothemBlock") {
                    0.125
                } else {
                    0.25
                };
                let vertical = match name {
                    "small_dripleaf" => 0.1,
                    "short_grass" | "fern" | "short_dry_grass" | "tall_dry_grass" => 0.2,
                    _ => 0.0,
                };
                (horizontal, vertical)
            });
            let mut face = [0u8; 6];
            let mut cover = [0f32; 6];
            for d in Direction::ALL {
                let (face_id, shape) = visibility
                    .occlusion_face(BlockStateId(id as u32), d)
                    .ok_or("missing occlusion face")?;
                face[d as usize] = face_id.min(2) as u8;
                cover[d as usize] = *covers
                    .entry((face_id, d as u8))
                    .or_insert_with(|| fluid::coverage(shape, d));
            }
            info.push(Info {
                solid: s.solid_render,
                full: s.full_collision,
                permeable: !s.solid_render || s.dampening == 0,
                shade,
                emission: s.emission,
                emissive,
                fluid: FluidState::from_block(state)?,
                model: s.render_shape == RenderShape::Model,
                block_entity: s.block_entity,
                tints,
                tint_count,
                offset,
                legacy_solid: s.legacy_solid,
                overlay: s.has_class("HalfTransparentBlock") || s.has_class("LeavesBlock"),
                leaves: s.has_class("LeavesBlock"),
                blocks_flow: assets.blocks_fluid_flow.contains(&*s.name),
                ice: s.has_class("IceBlock"),
                sturdy: s.sturdy_full,
                face,
                cover,
            });
        }
        Ok(Self {
            info,
            visibility,
            water_still: layer("minecraft:block/water_still")?,
            water_flow: layer("minecraft:block/water_flow")?,
            water_overlay: layer("minecraft:block/water_overlay")?,
            lava_still: layer("minecraft:block/lava_still")?,
            lava_flow: layer("minecraft:block/lava_flow")?,
        })
    }
    pub fn facts(&self, state: BlockStateId) -> Option<MeshFacts> {
        let i = self.info.get(state.0 as usize)?;
        Some(MeshFacts {
            shade: i.shade,
            emissive: i.emissive,
            legacy_solid: i.legacy_solid,
            light_permeable: i.permeable,
            solid_render: i.solid,
            full_collision: i.full,
            emission: i.emission,
            overlay: i.overlay,
            tint_sources: i.tint_count,
        })
    }
    /// `TerrainParticle`'s colour: tint source 0's `colorAsTerrainParticle`,
    /// which is the in-world (biome-blended) colour except on grass blocks,
    /// whose dirt crumbs stay untinted. `None` when the state has no source.
    /// `region[dz][dx]` are the chunks around `pos`'s chunk (centre 1,1).
    pub fn terrain_particle_tint(
        &self,
        state: BlockStateId,
        pos: [i32; 3],
        region: &[[Option<&Chunk>; 3]; 3],
        biomes: &Biomes,
        zoom: BiomeZoom,
        blend_radius: i32,
    ) -> Option<u32> {
        let info = self.info.get(state.0 as usize)?;
        if info.tints[0] == Tint::GrassBlock {
            return None;
        }
        let [x, y, z] = pos;
        let origin = [x >> 4, z >> 4, y & !15];
        let mut tints = Tints::new(region, biomes, zoom, blend_radius, origin);
        Mesher::tint(info, 0, &mut tints, [x & 15, y & 15, z & 15])
    }
    /// `BlockState.getOffset` for blocks with a random model offset.
    pub fn offset(&self, state: BlockStateId, x: i32, z: i32) -> Option<[f64; 3]> {
        offset(self.info.get(state.0 as usize)?, x, z)
    }
}
/// `OffsetType.XZ`/`XYZ`: a per-column jitter from `Mth.getSeed(x, 0, z)`.
fn offset(info: &Info, x: i32, z: i32) -> Option<[f64; 3]> {
    let (horizontal, vertical) = info.offset?;
    let seed = position_seed(BlockPos::new(x, 0, z));
    let nibble = |shift: u32| ((seed >> shift) & 15) as f32 / 15.0;
    let h = f64::from(horizontal);
    Some([
        ((f64::from(nibble(0)) - 0.5) * 0.5).clamp(-h, h),
        (f64::from(nibble(4)) - 1.0) * f64::from(vertical),
        ((f64::from(nibble(8)) - 0.5) * 0.5).clamp(-h, h),
    ])
}
/// Border width: smooth lighting samples up to two blocks out along a face normal.
const B: i32 = 2;
const P: usize = 16 + 2 * B as usize;
#[inline]
fn pi(x: i32, y: i32, z: i32) -> usize {
    ((y + B) as usize * P + (z + B) as usize) * P + (x + B) as usize
}
/// Blocks and light of one section plus a `B`-block border.
struct Padded {
    blocks: Vec<u32>,
    light: Vec<u8>,
}
impl Padded {
    fn new(region: &[[Option<&Chunk>; 3]; 3], sy: i32) -> Self {
        let mut blocks = vec![0u32; P * P * P];
        let mut light = vec![0u8; P * P * P];
        let base_y = MIN_Y + sy * 16;
        // The section itself is unpacked in one pass; the border cells of the
        // neighbouring sections are read one by one.
        let mut own = Box::new([0u32; 4096]);
        let center = region[1][1].map(|c| &c.sections[sy as usize]);
        if let Some(section) = center {
            section.unpack(&mut own);
        }
        for y in -B..16 + B {
            let wy = base_y + y;
            for z in -B..16 + B {
                for x in -B..16 + B {
                    let chunk = region[((z + 16) >> 4) as usize][((x + 16) >> 4) as usize];
                    let i = pi(x, y, z);
                    if let Some(c) = chunk {
                        blocks[i] =
                            if (0..16).contains(&x) && (0..16).contains(&y) && (0..16).contains(&z)
                            {
                                own[(y << 8 | z << 4 | x) as usize]
                            } else {
                                c.get(x & 15, wy, z & 15)
                            };
                        light[i] = c.light(x & 15, wy, z & 15);
                    } else {
                        light[i] = 0xF0;
                    }
                }
            }
        }
        Self { blocks, light }
    }
}
/// Set on a terrain vertex's sprite layer when its texture alpha must be ignored.
pub const OPAQUE_LAYER: u16 = 1 << 15;
/// The bits of `TerrainVertex::color`'s alpha byte that hold the light its face emits, for
/// HDR output's highlights (`gfx/hdr.rs`): 15 minus the level, so the usual 0xFF means none.
/// The byte's other four bits stay set.
pub const EMISSION_BITS: u32 = 0x0F00_0000;
/// `color` marked as emitting light `level` (0 to 15).
pub fn emitting(color: u32, level: u8) -> u32 {
    color & !EMISSION_BITS | u32::from(15 - level.min(15)) << 24
}
const UNSET: u32 = u32::MAX;
const NO_BIOME: u16 = u16::MAX;
/// The `ColorResolver`s `ClientLevel` caches block tints for.
#[derive(Clone, Copy)]
enum Resolver {
    Grass,
    Foliage,
    Dry,
    Water,
}
/// Biome tints of one block layer: the zoomed biome of every block of the
/// layer plus the blend radius around it, and the blended colours per column.
struct Layer {
    zoomed: Vec<u16>,
    colors: [[u32; 256]; 4],
}
/// `ClientLevel.calculateBlockTint` behind a per-layer `BlockTintCache`: each
/// block averages the colours of the biomes `BiomeManager.getBiome` picks (the
/// eight-corner zoom) over a `(2r+1)^2` square at the block's own height.
/// Layers and columns are only computed when a tinted block asks for them.
struct Tints<'a> {
    region: &'a [[Option<&'a Chunk>; 3]; 3],
    biomes: &'a Biomes,
    zoom: BiomeZoom,
    radius: i32,
    plains: u16,
    /// Chunk x, chunk z and the section's lowest block y.
    origin: [i32; 3],
    layers: [Option<Box<Layer>>; 16],
}
impl<'a> Tints<'a> {
    fn new(
        region: &'a [[Option<&'a Chunk>; 3]; 3],
        biomes: &'a Biomes,
        zoom: BiomeZoom,
        radius: i32,
        origin: [i32; 3],
    ) -> Self {
        Self {
            region,
            biomes,
            zoom,
            radius: radius.clamp(0, 7),
            plains: biomes.id("minecraft:plains"),
            origin,
            layers: Default::default(),
        }
    }
    /// The blended tint of block `(x, y, z)` (section-local x/z, y in 0..16).
    fn get(&mut self, resolver: Resolver, x: i32, y: i32, z: i32) -> u32 {
        let r = self.radius;
        let dim = (16 + 2 * r) as usize;
        let layer = self.layers[y as usize].get_or_insert_with(|| {
            Box::new(Layer {
                zoomed: vec![NO_BIOME; dim * dim],
                colors: [[UNSET; 256]; 4],
            })
        });
        let slot = (z * 16 + x) as usize;
        let cached = layer.colors[resolver as usize][slot];
        if cached != UNSET {
            return cached;
        }
        let [cx, cz, base_y] = self.origin;
        let (bx, by, bz) = (cx * 16, base_y + y, cz * 16);
        let (mut red, mut green, mut blue) = (0, 0, 0);
        for dz in -r..=r {
            for dx in -r..=r {
                let cell = ((z + dz + r) as usize) * dim + (x + dx + r) as usize;
                let mut biome = layer.zoomed[cell];
                if biome == NO_BIOME {
                    // BiomeManager.getBiome over the loaded chunks (plains
                    // where there is none).
                    let q = self
                        .zoom
                        .quart_at_block(BlockPos::new(bx + x + dx, by, bz + z + dz));
                    let (rx, rz) = ((q[0] >> 2) - cx + 1, (q[2] >> 2) - cz + 1);
                    biome = if (0..3).contains(&rx) && (0..3).contains(&rz) {
                        self.region[rz as usize][rx as usize].map_or(self.plains, |c| {
                            c.biome((q[0] & 3) * 4, q[1] * 4, (q[2] & 3) * 4)
                        })
                    } else {
                        self.plains
                    };
                    layer.zoomed[cell] = biome;
                }
                let c = match resolver {
                    Resolver::Grass => {
                        self.biomes
                            .grass(biome, f64::from(bx + x + dx), f64::from(bz + z + dz))
                    }
                    Resolver::Foliage => self.biomes.get(biome).foliage,
                    Resolver::Dry => self.biomes.get(biome).dry_foliage,
                    Resolver::Water => self.biomes.get(biome).water,
                };
                red += c >> 16 & 255;
                green += c >> 8 & 255;
                blue += c & 255;
            }
        }
        let n = ((2 * r + 1) * (2 * r + 1)) as u32;
        let color = (red / n) << 16 | (green / n) << 8 | (blue / n);
        layer.colors[resolver as usize][slot] = color;
        color
    }
}
/// Packs 8-bit RGB into the vertex's RGBA8 (red in the low byte), opaque.
fn pack_rgb([r, g, b]: [u32; 3]) -> u32 {
    0xFF00_0000 | b << 16 | g << 8 | r
}
/// Vanilla light coords (block*16 in the low half, sky*16 in the high half) to
/// the vertex layout (sky*16 in bits 0..8, block*16 in bits 8..16).
fn pack_coords(coords: u32) -> u16 {
    ((coords >> 16 & 0xFF) | (coords & 0xFF) << 8) as u16
}
pub struct Mesher<'a> {
    /// CardinalLighting.NETHER: brighter up/down faces.
    pub nether_shading: bool,
    pub assets: &'a Assets,
    pub render: &'a RenderAssets,
    pub tables: &'a MeshTables,
    pub biomes: &'a Biomes,
    pub visibility: FaceVisibility,
    /// The client's biome zoom (the hashed world seed `BiomeManager` holds).
    pub zoom: BiomeZoom,
    /// The `biomeBlendRadius` option, 0 to 7.
    pub blend_radius: i32,
    /// The `ao` option: smooth lighting for models that allow it.
    pub smooth_lighting: bool,
    /// The `cutoutLeaves` option (fancy graphics); off makes leaves opaque.
    pub cutout_leaves: bool,
    /// Native-renderer close-up greedy meshing. Disabled while a shader pack
    /// is active because third-party atlas shaders assume one sprite per quad.
    pub greedy_meshing: bool,
}
/// A single exposed side of an exact unit cube. Faces only share a key when
/// their material, projected UV transform, colour and light all agree, so a
/// merged quad cannot erase a texture, tint, AO or light boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GreedyKey {
    direction: u8,
    plane: u8,
    layer: u16,
    uv: [i8; 6],
    color: u32,
    light: u16,
}

type GreedyFaces = BTreeMap<GreedyKey, Vec<(u8, u8)>>;
type GreedyDirection = horizon_assets::model::Direction;

fn greedy_axes(direction: GreedyDirection) -> (usize, usize, usize) {
    match direction {
        GreedyDirection::Down | GreedyDirection::Up => (1, 0, 2),
        GreedyDirection::North | GreedyDirection::South => (2, 0, 1),
        GreedyDirection::West | GreedyDirection::East => (0, 2, 1),
    }
}

fn positive_face(direction: GreedyDirection) -> bool {
    matches!(
        direction,
        GreedyDirection::Up | GreedyDirection::South | GreedyDirection::East
    )
}

fn add_greedy_face(
    faces: &mut GreedyFaces,
    direction: GreedyDirection,
    pos: [i32; 3],
    style: crate::models::CubeStyle,
    vertices: &[TerrainVertex; 4],
) -> bool {
    // A rectangle has only four corners. Merge only flat-lit faces so all
    // original per-vertex attributes remain exactly representable. Also make
    // sure the baked face layer agrees with the cube classifier's source layer.
    if (vertices[0].layer & 0x3fff) != style.layer as u16 || vertices
        .iter()
        .any(|v| v.color != vertices[0].color || v.light != vertices[0].light)
    {
        return false;
    }
    let (axis, u_axis, v_axis) = greedy_axes(direction);
    let plane = pos[axis] + if positive_face(direction) { 1 } else { 0 };
    debug_assert!((0..=16).contains(&plane));
    let key = GreedyKey {
        direction: direction as u8,
        plane: plane as u8,
        layer: vertices[0].layer,
        uv: style.uv,
        color: vertices[0].color,
        light: vertices[0].light,
    };
    faces
        .entry(key)
        .or_default()
        .push((pos[u_axis] as u8, pos[v_axis] as u8));
    true
}

fn emit_greedy_rect(
    mesh: &mut SectionMesh,
    key: GreedyKey,
    u0: u8,
    v0: u8,
    width: u8,
    height: u8,
) {
    let direction = GreedyDirection::ALL[key.direction as usize];
    let (_axis, u_axis, v_axis) = greedy_axes(direction);
    let plane = f32::from(key.plane);
    let u0 = f32::from(u0);
    let v0 = f32::from(v0);
    let u1 = u0 + f32::from(width);
    let v1 = v0 + f32::from(height);
    let mut points = match direction {
        GreedyDirection::Down | GreedyDirection::Up => [
            [u0, plane, v1],
            [u1, plane, v1],
            [u1, plane, v0],
            [u0, plane, v0],
        ],
        GreedyDirection::North | GreedyDirection::South => [
            [u0, v1, plane],
            [u1, v1, plane],
            [u1, v0, plane],
            [u0, v0, plane],
        ],
        GreedyDirection::West | GreedyDirection::East => [
            [plane, v1, u1],
            [plane, v1, u0],
            [plane, v0, u0],
            [plane, v0, u1],
        ],
    };
    if matches!(
        direction,
        GreedyDirection::Down | GreedyDirection::South | GreedyDirection::East
    ) {
        points.reverse();
    }

    // CubeStyle maps the spatial (u,v) face coordinates to the source
    // sprite. Extending that affine mapping across blocks repeats the sprite
    // once per block, including mirrored and quarter-turned variants.
    let [a, b, du, dv, eu, ev] = key.uv.map(f32::from);
    let mut uvs = points.map(|point| {
        let face_u = point[u_axis] - u0;
        let face_v = point[v_axis] - v0;
        [a + du * face_u + eu * face_v, b + dv * face_u + ev * face_v]
    });
    let origin = [0, 1].map(|i| {
        uvs.iter()
            .map(|uv| uv[i])
            .fold(f32::INFINITY, f32::min)
            .floor()
    });
    for uv in &mut uvs {
        uv[0] -= origin[0];
        uv[1] -= origin[1];
    }
    mesh.solid.extend(points.into_iter().zip(uvs).map(|(pos, uv)| {
        TerrainVertex::tiled_section(pos, uv, key.layer, key.color, key.light)
    }));
}

/// Greedily covers each coplanar material/light-compatible face mask with
/// maximal rectangles. One cell represents one block side; the rectangle's
/// UVs repeat per block and its vertices retain the exact common attributes.
fn emit_greedy_faces(mesh: &mut SectionMesh, faces: GreedyFaces) {
    for (key, cells) in faces {
        let mut mask = [[false; 16]; 16];
        for (u, v) in cells {
            mask[usize::from(v)][usize::from(u)] = true;
        }
        for v in 0..16usize {
            for u in 0..16usize {
                if !mask[v][u] {
                    continue;
                }
                let mut width = 1;
                while u + width < 16 && mask[v][u + width] {
                    width += 1;
                }
                let mut height = 1;
                'rows: while v + height < 16 {
                    for x in u..u + width {
                        if !mask[v + height][x] {
                            break 'rows;
                        }
                    }
                    height += 1;
                }
                for row in &mut mask[v..v + height] {
                    row[u..u + width].fill(false);
                }
                emit_greedy_rect(mesh, key, u as u8, v as u8, width as u8, height as u8);
            }
        }
    }
}

impl Mesher<'_> {
    /// `region[dz][dx]` are the chunks around the meshed column (centre 1,1).
    pub fn mesh(&mut self, region: &[[Option<&Chunk>; 3]; 3], sy: i32) -> SectionMesh {
        let mut out = SectionMesh::default();
        let center = region[1][1].expect("centre chunk");
        let section = &center.sections[sy as usize];
        if section.is_uniform(0) {
            return out;
        }
        let pad = Padded::new(region, sy);
        out.visibility = visibility(&pad, &self.tables.info);
        let base_y = MIN_Y + sy * 16;
        let mut tints = Tints::new(
            region,
            self.biomes,
            self.zoom,
            self.blend_radius,
            [center.x, center.z, base_y],
        );
        let info = &self.tables.info;
        let catalog = &self.render.model_catalog;
        let mut models = Vec::new();
        let mut greedy_faces = GreedyFaces::new();
        for y in 0..16 {
            for z in 0..16 {
                for x in 0..16 {
                    let id = pad.blocks[pi(x, y, z)];
                    if id == 0 {
                        continue;
                    }
                    let state = &info[id as usize];
                    if state.block_entity {
                        out.block_entities
                            .push(([center.x * 16 + x, base_y + y, center.z * 16 + z], id));
                    }
                    if state.solid
                        && Direction::ALL.iter().all(|d| {
                            let [dx, dy, dz] = d.normal();
                            info[pad.blocks[pi(x + dx, y + dy, z + dz)] as usize].solid
                        })
                    {
                        continue;
                    }
                    if state.fluid.kind() != FluidKind::Empty {
                        let water = tints.get(Resolver::Water, x, y, z);
                        let nether = self.nether_shading;
                        fluid::tesselate(self.tables, nether, &pad, water, &mut out, [x, y, z]);
                    }
                    if !state.model {
                        continue;
                    }
                    let world = BlockPos::new(center.x * 16 + x, base_y + y, center.z * 16 + z);
                    if catalog
                        .models_at(BlockStateId(id), world, &mut models)
                        .is_err()
                    {
                        continue;
                    }
                    // ModelBlockRenderer: the first part decides smooth lighting
                    // for all; light-emitting blocks are always lit flat.
                    let Some(first) = models.first() else {
                        continue;
                    };
                    let ao = self.smooth_lighting && state.emission == 0 && first.ambient_occlusion;
                    let block_offset = offset(state, world.x, world.z);
                    let origin = match block_offset {
                        Some(o) => [
                            x as f32 + o[0] as f32,
                            y as f32 + o[1] as f32,
                            z as f32 + o[2] as f32,
                        ],
                        None => [x as f32, y as f32, z as f32],
                    };
                    let mut visible = [false; 6];
                    for d in Direction::ALL {
                        let [dx, dy, dz] = d.normal();
                        let neighbor = pad.blocks[pi(x + dx, y + dy, z + dz)];
                        let other = &info[neighbor as usize];
                        // LeavesBlock.skipRendering with opaque leaves.
                        let opaque_leaves = !self.cutout_leaves && state.leaves && other.leaves;
                        visible[d as usize] = !opaque_leaves
                            && self
                                .visibility
                                .should_render_face(
                                    BlockStateId(id),
                                    BlockStateId(neighbor),
                                    d,
                                    true,
                                )
                                .unwrap_or(true);
                    }
                    // Only exact, unshifted, opaque unit cubes are candidates.
                    // Multi-part models, fluids, animated/offset blocks,
                    // tinted or emissive materials retain their original path.
                    let cube = if self.greedy_meshing
                        && state.solid
                        && state.fluid.kind() == FluidKind::Empty
                        && state.emission == 0
                        && !state.emissive
                        && block_offset.is_none()
                        && models.len() == 1
                    {
                        models[0].cube
                    } else {
                        None
                    };
                    if let Some(styles) = cube {
                        for face in &models[0].faces {
                            let Some(direction) = face.quad.cull else {
                                // `cube_styles` guarantees this cannot occur;
                                // keep a defensive fallback if the catalog changes.
                                self.face(
                                    &pad, &mut tints, &mut out, [x, y, z], origin, face, ao, state,
                                );
                                continue;
                            };
                            if !visible[direction as usize] {
                                continue;
                            }
                            let vertices = self.face_vertices(
                                &pad,
                                &mut tints,
                                [x, y, z],
                                origin,
                                face,
                                ao,
                                state,
                            );
                            if !add_greedy_face(
                                &mut greedy_faces,
                                direction,
                                [x, y, z],
                                styles[direction as usize],
                                &vertices,
                            ) {
                                // Non-uniform corner lighting cannot be
                                // represented by one larger quad.
                                out.solid.extend(vertices);
                            }
                        }
                    } else {
                        for model in &models {
                            for face in &model.faces {
                                if face.quad.cull.is_some_and(|d| !visible[d as usize]) {
                                    continue;
                                }
                                self.face(
                                    &pad,
                                    &mut tints,
                                    &mut out,
                                    [x, y, z],
                                    origin,
                                    face,
                                    ao,
                                    state,
                                );
                            }
                        }
                    }
                }
            }
        }
        emit_greedy_faces(&mut out, greedy_faces);
        out
    }
    /// `BlockColors` tint source `index`, or `None` for an untinted (-1) layer.
    fn tint(state: &Info, index: i32, tints: &mut Tints, [x, y, z]: [i32; 3]) -> Option<u32> {
        if index < 0 || index >= i32::from(state.tint_count) {
            return None;
        }
        Some(match state.tints[(index as usize).min(1)] {
            Tint::None => return None,
            Tint::Grass | Tint::GrassBlock => tints.get(Resolver::Grass, x, y, z),
            Tint::TallGrassUpper => tints.get(Resolver::Grass, x, y, z),
            Tint::Foliage => tints.get(Resolver::Foliage, x, y, z),
            Tint::DryFoliage => tints.get(Resolver::Dry, x, y, z),
            Tint::Water => tints.get(Resolver::Water, x, y, z),
            Tint::Constant(c) => c,
            Tint::Redstone(p) => redstone_color(p),
            Tint::Stem(age) => {
                let a = u32::from(age);
                (a * 32) << 16 | (255 - a * 8) << 8 | (a * 4)
            }
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn face(
        &self,
        pad: &Padded,
        tints: &mut Tints,
        out: &mut SectionMesh,
        pos: [i32; 3],
        origin: [f32; 3],
        face: &BakedFace,
        ao: bool,
        state: &Info,
    ) {
        let vertices = self.face_vertices(pad, tints, pos, origin, face, ao, state);
        if face.translucent {
            out.translucent.extend(vertices);
        } else {
            out.solid.extend(vertices);
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn face_vertices(
        &self,
        pad: &Padded,
        tints: &mut Tints,
        pos: [i32; 3],
        origin: [f32; 3],
        face: &BakedFace,
        ao: bool,
        state: &Info,
    ) -> [TerrainVertex; 4] {
        let q = &face.quad;
        let shade_face = q.shade_direction_override.unwrap_or(q.facing);
        let directional = lighting::cardinal(self.nether_shading, shade_face);
        // Grey level per vertex before tinting, and vanilla light coords.
        let (gray, coords) = if ao {
            let info = &self.tables.info;
            let (brightness, coords) =
                lighting::ambient_occlusion(pad, info, state, pos, &q.positions, q.facing);
            let gray = brightness.map(|b| lighting::scale(lighting::gray(b), directional));
            (gray, coords)
        } else {
            let coords = match q.cull {
                Some(d) => lighting::coords(pad, state, lighting::step(pos, d)),
                None => lighting::flat(pad, state, pos, &q.positions, q.facing),
            };
            ([lighting::gray(directional); 4], [coords; 4])
        };
        let tint = Self::tint(state, q.tint_index, tints, pos);
        std::array::from_fn(|i| {
            let g = gray[i] as u32;
            let rgb = match tint {
                Some(t) => [16, 8, 0].map(|shift| g * (t >> shift & 255) / 255),
                None => [g; 3],
            };
            let p = q.positions[i];
            TerrainVertex::new(
                [p[0] + origin[0], p[1] + origin[1], p[2] + origin[2]],
                q.uv[i],
                // Opaque leaves (forceOpaque) and quads over opaque texels draw in the
                // solid layer: no alpha cutout (a dirt path's side skips its clear top row).
                if face.opaque && !face.translucent || !self.cutout_leaves && state.leaves {
                    face.layer as u16 | OPAQUE_LAYER
                } else {
                    face.layer as u16
                },
                emitting(pack_rgb(rgb), state.emission.max(q.light_emission)),
                pack_coords(lighting::with_emission(coords[i], q.light_emission)),
            )
        })
    }
}
/// Snapshot of the 3x3 chunk neighbourhood used by a worker job.
pub type Region = [[Option<Arc<Chunk>>; 3]; 3];
pub fn borrow(region: &Region) -> [[Option<&Chunk>; 3]; 3] {
    std::array::from_fn(|z| std::array::from_fn(|x| region[z][x].as_deref()))
}

#[cfg(test)]
mod greedy_meshing_tests {
    use super::*;

    fn key(direction: GreedyDirection, layer: u16, color: u32, light: u16) -> GreedyKey {
        GreedyKey {
            direction: direction as u8,
            plane: if positive_face(direction) { 1 } else { 0 },
            layer,
            uv: [0, 0, 1, 0, 0, 1],
            color,
            light,
        }
    }

    fn tiled_uv(vertex: &TerrainVertex) -> [f32; 2] {
        vertex.uv.map(|v| f32::from(v) / TILED_UV_STEPS)
    }

    #[test]
    fn same_material_and_light_faces_merge_and_repeat_per_block() {
        let mut faces = GreedyFaces::new();
        let key = key(GreedyDirection::Up, 7, u32::MAX, 0xF0);
        faces
            .entry(key)
            .or_default()
            .extend((0..4).flat_map(|v| (0..4).map(move |u| (u, v))));

        let mut mesh = SectionMesh::default();
        emit_greedy_faces(&mut mesh, faces);

        assert_eq!(mesh.solid.len(), 4, "a 4x4 surface is one quad");
        assert!(mesh.translucent.is_empty());
        assert!(mesh.solid.iter().all(|v| v.layer & TILED_UV_LAYER != 0));
        assert!(mesh.solid.iter().all(|v| v.layer & 0x3fff == 7));
        let uv = mesh.solid.iter().map(tiled_uv).collect::<Vec<_>>();
        let range = |axis: usize| {
            uv.iter().map(|v| v[axis]).fold(f32::NEG_INFINITY, f32::max)
                - uv.iter().map(|v| v[axis]).fold(f32::INFINITY, f32::min)
        };
        assert_eq!([range(0), range(1)], [4.0, 4.0]);
        let points = mesh.solid.iter().map(|v| v.pos).collect::<Vec<_>>();
        assert!(points.iter().all(|p| p[1] == 1.0));
    }

    #[test]
    fn material_and_light_boundaries_prevent_merging() {
        let mut faces = GreedyFaces::new();
        faces.entry(key(GreedyDirection::Up, 3, u32::MAX, 0xF0)).or_default().push((0, 0));
        faces.entry(key(GreedyDirection::Up, 4, u32::MAX, 0xF0)).or_default().push((1, 0));
        faces.entry(key(GreedyDirection::Up, 3, u32::MAX, 0x10)).or_default().push((2, 0));
        let mut mesh = SectionMesh::default();
        emit_greedy_faces(&mut mesh, faces);
        assert_eq!(mesh.solid.len(), 12, "three incompatible faces remain separate");
    }

    #[test]
    fn varying_corner_lighting_uses_the_original_quad() {
        let mut vertices = [TerrainVertex::new([0.0; 3], [0.0; 2], 1, 10, 20); 4];
        vertices[2].color = 11;
        let mut faces = GreedyFaces::new();
        assert!(!add_greedy_face(
            &mut faces,
            GreedyDirection::Up,
            [0, 0, 0],
            crate::models::CubeStyle {
                layer: 1,
                uv: [0, 0, 1, 0, 0, 1],
                shade_bits: 1.0f32.to_bits(),
            },
            &vertices,
        ));
        assert!(faces.is_empty());
    }

    #[test]
    fn greedy_rectangles_keep_each_direction_winding() {
        for direction in GreedyDirection::ALL {
            let mut mesh = SectionMesh::default();
            emit_greedy_rect(&mut mesh, key(direction, 1, u32::MAX, 240), 2, 3, 2, 3);
            let q: [TerrainVertex; 4] = mesh.solid.clone().try_into().unwrap();
            let p = |i: usize| glam::Vec3::from(q[i].pos);
            let normal = (p(1) - p(0)).cross(p(2) - p(0));
            let wanted = glam::Vec3::from(match direction {
                GreedyDirection::Down => [0.0, -1.0, 0.0],
                GreedyDirection::Up => [0.0, 1.0, 0.0],
                GreedyDirection::North => [0.0, 0.0, -1.0],
                GreedyDirection::South => [0.0, 0.0, 1.0],
                GreedyDirection::West => [-1.0, 0.0, 0.0],
                GreedyDirection::East => [1.0, 0.0, 0.0],
            });
            assert!(normal.dot(wanted) > 0.0, "{direction:?}: {normal:?}");
        }
    }
}
