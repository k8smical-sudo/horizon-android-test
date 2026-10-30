# Close-up terrain greedy meshing

The section mesher now merges compatible faces of exact, opaque unit-cube
models before the section mesh is uploaded. The distant LOD mesher is unchanged.

## What is eligible

A block is considered only when it has one baked model whose six faces pass the
existing `cube_styles` exact-cube test, is solid and non-fluid, has no random
model offset, and does not emit light. Each visible face is merged only when
its material layer, projected UV transform, baked vertex colour and packed
light are identical. Faces with varying corner AO/light remain as their
original four-vertex quads. Slabs, stairs, plants, fluids, translucent/cutout
surfaces, tinted/emissive models and multipart models retain the existing
mesher path.

The greedy scan is section-local and operates independently for each face
orientation and plane. It builds maximal rectangles from the compatible face
mask. UVs repeat once per block using the tiled UV encoding; a reserved layer
bit selects that encoding in the native terrain and native material shaders.
The vertex layout itself is unchanged.

Third-party shader packs use an atlas-coordinate path that assumes one sprite
per quad. Greedy meshing is therefore disabled while a shader pack is selected
and loaded sections are queued for remeshing when switching into or out of that
mode.

## Checks

Unit tests were added for rectangle merging and per-block UV repetition,
material/light boundaries, non-uniform-corner fallback, and winding on all six
face directions. Run them on a Rust toolchain with:

```sh
cargo fmt --all
cargo test -p horizon-client --lib mesher::greedy_meshing_tests
cargo test -p horizon-client --lib --bins --tests
cargo build --profile fast -p horizon-client
```

The source-editing environment used to prepare this archive did not have
`cargo` or `rustc` installed, so these Rust checks have not been executed here.
