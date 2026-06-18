# Primitive Showcase

These JSON files document small primitive presets used by the USER demos.
They are intentionally simple: each file names a primitive, basic dimensions,
material properties, and the demo path that exercises it.

The live poly fruit bowl uses the same concepts directly through `py_3d`
objects: `Bowl`, generated low-poly fruit, `HangingConeLampPrimitive`, `Lamp`,
`FloatingTextBulletin`, and fixed sign geometry.

Prepared imported assets live separately under `USER/assets/`. Use
`examples/ingest_asset.py` to convert OBJ files into py_3d mesh assets, then
exercise them through dedicated render demos such as
`USER/demos/14_render_sea_lion_asset.py`.
