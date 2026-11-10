# Logo build

[`build_logo.py`](build_logo.py) builds Basir's two marks from the room it reconstructed in Phase 0:

- **Logo** (`logo/basir-logo-*`): the scanned room as isometric voxels, a two-bubble chat (your question, Basir's answer with the direction) and a pin on the found object.
- **README image** (`logo/basir-readme-*`): the same scan on a phone screen, with the conversation beside it.

```bash
uv run python scripts/visualize_pointcloud.py fr1/room --no-window   # once, to create the scan
uv run python logo/src/build_logo.py                                  # rebuild logo/*.svg and *.png
```

Each mark is written as light and dark SVG (colours baked in, the scan embedded) plus PNG sizes, rasterised with headless Google Chrome. `--no-png` writes the SVGs only, and `--out DIR` writes somewhere else.
