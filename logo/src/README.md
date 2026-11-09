# Logo generators

These scripts build Basir's marks from the room it reconstructed in Phase 0 (`outputs/rgbd_dataset_freiburg1_room.ply`, created by `scripts/visualize_pointcloud.py fr1/room`).

| Script | Does what |
|---|---|
| `export.py` | Writes the chosen marks to `logo/`: the logo (scan + chat) and the README image (scan on a phone), as light / dark SVG and PNG |
| `generate4.py` | The four scan + chat compositions (conversation, eye badge, phone chat, answer chip) as a design board |
| `generate3.py` | The three 3D treatments of the scan (voxel blocks, corner perspective, exploded layers) |
| `generate.py` | The modelled room and apartment concepts, and the shared speech bubble |

```bash
uv run python scripts/visualize_pointcloud.py fr1/room --no-window   # once, to create the scan
cd logo/src && uv run python export.py ../..                          # rebuild logo/*.svg and *.png
```

`export.py` rasterises the PNGs with headless Google Chrome. The design-board scripts write `index.html` next to themselves (git-ignored).
