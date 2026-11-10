"""The TUM RGB-D benchmark catalogue: preset keys, download URLs and name resolution."""

from __future__ import annotations

# TUM moved the benchmark to cvg.cit.tum.de; vision.in.tum.de only redirects there.
TUM_BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset"

# key -> (sequence name, description). Sizes measured from the server.
_TUM_SEQUENCES: dict[str, tuple[str, str]] = {
    "fr1/room": ("rgbd_dataset_freiburg1_room", "~746 MB, handheld traversal of a whole office room"),
    "fr1/desk": ("rgbd_dataset_freiburg1_desk", "~328 MB, desk with monitor, mugs, books"),
    "fr1/desk2": ("rgbd_dataset_freiburg1_desk2", "~333 MB, second desk sequence"),
    "fr1/xyz": ("rgbd_dataset_freiburg1_xyz", "~427 MB, translation along 3 principal axes"),
    "fr1/rpy": ("rgbd_dataset_freiburg1_rpy", "~391 MB, rotation around roll, pitch, yaw"),
    "fr1/plant": ("rgbd_dataset_freiburg1_plant", "~663 MB, house plant on a table"),
    "fr1/teddy": ("rgbd_dataset_freiburg1_teddy", "~838 MB, teddy bear on an office chair"),
    "fr1/360": ("rgbd_dataset_freiburg1_360", "~399 MB, 360 degree turn in the room"),
    "fr1/floor": ("rgbd_dataset_freiburg1_floor", "~701 MB, floor loop trajectory"),
    "fr2/desk": ("rgbd_dataset_freiburg2_desk", "~1.8 GB, office desk, calibrated camera"),
    "fr2/xyz": ("rgbd_dataset_freiburg2_xyz", "~2.1 GB, translation along 3 axes"),
    "fr2/rpy": ("rgbd_dataset_freiburg2_rpy", "~1.9 GB, rotation along 3 axes"),
    "fr2/large_with_loop": ("rgbd_dataset_freiburg2_large_with_loop", "~2.7 GB, industrial hall with loop closure"),
    "fr3/long_office": ("rgbd_dataset_freiburg3_long_office_household", "~1.4 GB, long office household loop"),
    "fr3/nostructure_texture_near": ("rgbd_dataset_freiburg3_nostructure_texture_near_withloop", "~745 MB"),
    "fr3/structure_texture_far": ("rgbd_dataset_freiburg3_structure_texture_far", "~497 MB"),
    "fr3/sitting_xyz": ("rgbd_dataset_freiburg3_sitting_xyz", "~739 MB, dynamic scene, people sitting"),
    "fr3/walking_xyz": ("rgbd_dataset_freiburg3_walking_xyz", "~503 MB, dynamic scene, people walking"),
}


def _preset(name: str, description: str) -> dict[str, str]:
    group = name.split("_")[2]  # rgbd_dataset_freiburg1_room -> freiburg1
    return {"url": f"{TUM_BASE_URL}/{group}/{name}.tgz", "name": name, "description": description}


PRESETS: dict[str, dict[str, str]] = {key: _preset(name, desc) for key, (name, desc) in _TUM_SEQUENCES.items()}


def _aliases(key: str, name: str) -> set[str]:
    """fr1/room -> fr1_room, tum_fr1_room, freiburg1_room, rgbd_dataset_freiburg1_room."""
    flat = key.replace("/", "_")
    return {flat, f"tum_{flat}", name, name.removeprefix("rgbd_dataset_")}


PRESET_ALIASES: dict[str, str] = {alias: key for key, info in PRESETS.items() for alias in _aliases(key, info["name"])}


def resolve_preset_key(identifier: str | None) -> str | None:
    """Canonical preset key for a key or alias (case, '-', ' ' and '/' are normalised)."""
    if not identifier:
        return None
    raw = identifier.strip().lower()
    if raw in PRESETS:
        return raw
    return PRESET_ALIASES.get(raw.replace("-", "_").replace(" ", "_").replace("/", "_"))


def get_preset_info(identifier: str | None) -> dict[str, str] | None:
    key = resolve_preset_key(identifier)
    return PRESETS[key] if key else None
