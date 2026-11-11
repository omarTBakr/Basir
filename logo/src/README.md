# Logo build

[`build_logo.py`](build_logo.py) builds Basir's logo (`logo/basir-logo-*`) from the room it reconstructed in Phase 0: the scanned room as isometric voxels, a two-bubble chat (your question, Basir's answer with the direction) and a pin on the found object.

```bash
uv run python scripts/visualize_pointcloud.py fr1/room --no-window   # once, to create the scan
uv run python logo/src/build_logo.py                                  # rebuild logo/*.svg and *.png
```

It is written as light and dark SVG (colours baked in, the scan embedded) plus PNG sizes, rasterised with headless Google Chrome. `--no-png` writes the SVGs only, and `--out DIR` writes somewhere else.
