"""Asset manifest and resource management for the OpenGL renderer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .context import HeadlessContextError
from .gpu import (
    GLBindings,
    GPUBuffer,
    Texture2D,
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_NEAREST,
    GL_REPEAT,
    GL_STATIC_DRAW,
)

__all__ = [
    "AssetStore",
    "SceneManifest",
    "SceneNode",
]


_LOGGER = logging.getLogger(__name__)

_DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
_DEFAULT_MANIFEST_NAME = "sim_scene.yaml"


# --------------------------------------------------------------------------- dataclasses


@dataclass
class TextureDefinition:
    name: str
    path: Path
    srgb: bool = True
    generate_mipmaps: bool = True
    wrap_s: str = "clamp"
    wrap_t: str = "clamp"
    min_filter: str = "linear"
    mag_filter: str = "linear"


@dataclass
class MaterialDefinition:
    name: str
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    texture: Optional[str] = None
    double_sided: bool = False


@dataclass
class TransformDefinition:
    translate: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class SceneNode:
    name: str
    mesh: str
    transform: TransformDefinition = field(default_factory=TransformDefinition)
    tags: Tuple[str, ...] = ()
    animation: Optional[Dict[str, Any]] = None


@dataclass
class MeshDefinition:
    name: str
    path: Optional[Path] = None
    primitive: Optional[str] = None
    material: Optional[str] = None
    size: Tuple[float, float] = (1.0, 1.0)
    up_axis: str = "y"
    scale: Optional[Tuple[float, float, float]] = None
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class MeshGeometry:
    vertices: np.ndarray
    indices: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


@dataclass
class MeshResource:
    name: str
    material: Optional[str]
    geometry: MeshGeometry
    vertex_buffer: GPUBuffer
    index_buffer: Optional[GPUBuffer]


@dataclass
class SceneManifest:
    root: Path
    source: Optional[Path]
    textures: Dict[str, TextureDefinition] = field(default_factory=dict)
    materials: Dict[str, MaterialDefinition] = field(default_factory=dict)
    meshes: Dict[str, MeshDefinition] = field(default_factory=dict)
    static_nodes: List[SceneNode] = field(default_factory=list)
    target_nodes: List[SceneNode] = field(default_factory=list)
    generated: bool = False

    # ---------------------------------------------------------------- factory helpers
    @classmethod
    def from_file(cls, path: Path, root: Path) -> "SceneManifest":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        manifest = cls(root=root, source=path)
        manifest._parse(data)
        return manifest

    @classmethod
    def default(cls, root: Path) -> "SceneManifest":
        manifest = cls(root=root, source=None, generated=True)

        textures = {}
        for name, filename in ("drone", "drone.png"), ("person", "person.png"):
            tex_path = root / filename
            if tex_path.exists():
                textures[name] = TextureDefinition(name=name, path=tex_path)
        manifest.textures = textures

        materials: Dict[str, MaterialDefinition] = {
            "ground": MaterialDefinition(name="ground", color=(0.32, 0.36, 0.42)),
        }
        if "drone" in textures:
            materials["drone"] = MaterialDefinition(
                name="drone", color=(0.8, 0.8, 0.8), texture="drone"
            )
        if "person" in textures:
            materials["person"] = MaterialDefinition(
                name="person", color=(1.0, 1.0, 1.0), texture="person"
            )
        manifest.materials = materials

        meshes: Dict[str, MeshDefinition] = {
            "ground": MeshDefinition(
                name="ground",
                primitive="plane",
                material="ground",
                size=(40.0, 40.0),
                up_axis="y",
                offset=(0.0, -0.01, 0.0),
            ),
        }

        drone_obj = root / "drone.obj"
        if drone_obj.exists():
            meshes["drone"] = MeshDefinition(
                name="drone",
                path=drone_obj,
                material="drone" if "drone" in materials else None,
            )
        person_obj = root / "person.obj"
        if person_obj.exists():
            meshes["person"] = MeshDefinition(
                name="person",
                path=person_obj,
                material="person" if "person" in materials else None,
            )
        manifest.meshes = meshes

        manifest.static_nodes = [
            SceneNode(
                name="ground",
                mesh="ground",
                transform=TransformDefinition(),
            )
        ]

        target_nodes: List[SceneNode] = []
        if "drone" in meshes:
            target_nodes.append(
                SceneNode(
                    name="drone_target",
                    mesh="drone",
                    transform=TransformDefinition(translate=(0.0, 0.0, 6.5)),
                )
            )
        if "person" in meshes:
            target_nodes.append(
                SceneNode(
                    name="person_target",
                    mesh="person",
                    transform=TransformDefinition(translate=(-2.5, 0.0, 8.0)),
                )
            )
        manifest.target_nodes = target_nodes

        return manifest

    # ------------------------------------------------------------------ parsing
    def _parse(self, data: Dict[str, Any]) -> None:
        textures = data.get("textures", [])
        self.textures = {
            tex.name: tex
            for tex in (_parse_texture(self.root, entry) for entry in _iter_dicts(textures))
            if tex is not None
        }

        materials = data.get("materials", [])
        self.materials = {
            mat.name: mat
            for mat in (_parse_material(entry) for entry in _iter_dicts(materials))
            if mat is not None
        }

        meshes = data.get("meshes", [])
        self.meshes = {
            mesh.name: mesh
            for mesh in (_parse_mesh(self.root, entry) for entry in _iter_dicts(meshes))
            if mesh is not None
        }

        scene = data.get("scene", {})
        self.static_nodes = [
            node
            for node in (_parse_scene_node(entry) for entry in _iter_dicts(scene.get("static", [])))
            if node is not None
        ]
        self.target_nodes = [
            node
            for node in (_parse_scene_node(entry) for entry in _iter_dicts(scene.get("targets", [])))
            if node is not None
        ]


# --------------------------------------------------------------------------- asset store


class AssetStore:
    """Load meshes, textures, and materials into GPU resources."""

    def __init__(
        self,
        bindings: GLBindings,
        config: Optional[Dict[str, Any]] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._gl = bindings
        self._logger = logger or _LOGGER
        self._config = dict(config or {})

        self.root_path = self._resolve_root(self._config)
        manifest_path = self._resolve_manifest_path(self._config, self.root_path)
        self.manifest = self._load_manifest(manifest_path)

        self.textures: Dict[str, Texture2D] = {}
        self.materials = self.manifest.materials
        self.meshes: Dict[str, MeshResource] = {}

        self._load_textures()
        self._load_meshes()

    # ------------------------------------------------------------------ cleanup
    def close(self) -> None:
        for mesh in list(self.meshes.values()):
            if mesh.index_buffer is not None:
                mesh.index_buffer.release()
            mesh.vertex_buffer.release()
        self.meshes.clear()

        for tex in list(self.textures.values()):
            tex.release()
        self.textures.clear()

    # ----------------------------------------------------------------- internals
    def _resolve_root(self, cfg: Dict[str, Any]) -> Path:
        raw_root = cfg.get("root")
        if isinstance(raw_root, str) and raw_root.strip():
            root = Path(raw_root)
            if not root.is_absolute():
                root = (_DEFAULT_ASSET_ROOT.parent / root).resolve()
            return root
        return _DEFAULT_ASSET_ROOT

    def _resolve_manifest_path(self, cfg: Dict[str, Any], root: Path) -> Path:
        raw_manifest = cfg.get("manifest")
        if isinstance(raw_manifest, str) and raw_manifest.strip():
            manifest = Path(raw_manifest)
            if not manifest.is_absolute():
                manifest = (root / manifest).resolve()
            return manifest
        return (root / _DEFAULT_MANIFEST_NAME).resolve()

    def _load_manifest(self, path: Path) -> SceneManifest:
        if path.exists():
            try:
                manifest = SceneManifest.from_file(path, self.root_path)
                self._logger.info(
                    "Loaded scene manifest: %s (meshes=%d, textures=%d)",
                    path,
                    len(manifest.meshes),
                    len(manifest.textures),
                )
                return manifest
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning("Failed to parse manifest '%s': %s", path, exc)

        self._logger.warning(
            "Using default asset manifest; unable to load '%s'", path
        )
        return SceneManifest.default(self.root_path)

    def _load_textures(self) -> None:
        for name, definition in self.manifest.textures.items():
            try:
                image = _load_texture_pixels(definition.path)
            except Exception as exc:
                self._logger.warning("Texture '%s' failed to load: %s", name, exc)
                continue

            texture = self._gl.create_texture2d(
                image,
                srgb=definition.srgb,
                generate_mipmaps=definition.generate_mipmaps,
                min_filter=_resolve_filter(definition.min_filter),
                mag_filter=_resolve_filter(definition.mag_filter),
                wrap_s=_resolve_wrap(definition.wrap_s),
                wrap_t=_resolve_wrap(definition.wrap_t),
            )
            self.textures[name] = texture
            self._logger.debug(
                "Texture '%s' uploaded (%dx%d, srgb=%s)",
                name,
                texture.width,
                texture.height,
                definition.srgb,
            )

    def _load_meshes(self) -> None:
        for name, definition in self.manifest.meshes.items():
            try:
                geometry = _build_geometry(definition)
            except Exception as exc:
                self._logger.warning("Mesh '%s' failed to load: %s", name, exc)
                continue

            if definition.scale is not None:
                scale = np.asarray(definition.scale, dtype=np.float32).reshape(3)
                geometry.vertices[:, 0:3] *= scale
            if definition.offset != (0.0, 0.0, 0.0):
                offset = np.asarray(definition.offset, dtype=np.float32).reshape(3)
                geometry.vertices[:, 0:3] += offset

            vertex_buffer = self._gl.create_buffer(GL_ARRAY_BUFFER, geometry.vertices)

            if geometry.indices.size > 0:
                index_buffer = self._gl.create_buffer(
                    GL_ELEMENT_ARRAY_BUFFER, geometry.indices, usage=GL_STATIC_DRAW
                )
            else:
                index_buffer = None

            self.meshes[name] = MeshResource(
                name=name,
                material=definition.material,
                geometry=geometry,
                vertex_buffer=vertex_buffer,
                index_buffer=index_buffer,
            )

            self._logger.debug(
                "Mesh '%s' uploaded (verts=%d, tris=%d)",
                name,
                geometry.vertices.shape[0],
                geometry.indices.size // 3,
            )


# --------------------------------------------------------------------------- helpers


def _iter_dicts(values: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, Iterable):
        return []
    return (item for item in values if isinstance(item, dict))


def _parse_texture(root: Path, data: Dict[str, Any]) -> Optional[TextureDefinition]:
    name = str(data.get("name", "")).strip()
    if not name:
        return None
    path_value = data.get("path") or data.get("file")
    if not isinstance(path_value, str):
        return None
    texture_path = Path(path_value)
    if not texture_path.is_absolute():
        texture_path = (root / texture_path).resolve()

    return TextureDefinition(
        name=name,
        path=texture_path,
        srgb=bool(data.get("srgb", True)),
        generate_mipmaps=bool(data.get("mipmaps", True)),
        wrap_s=str(data.get("wrap_s", data.get("wrap", "clamp"))).lower(),
        wrap_t=str(data.get("wrap_t", data.get("wrap", "clamp"))).lower(),
        min_filter=str(data.get("min_filter", "linear")).lower(),
        mag_filter=str(data.get("mag_filter", "linear")).lower(),
    )


def _parse_material(data: Dict[str, Any]) -> Optional[MaterialDefinition]:
    name = str(data.get("name", "")).strip()
    if not name:
        return None

    color = _coerce_vector(data.get("color"), 3, default=(1.0, 1.0, 1.0))
    texture = data.get("texture")
    if isinstance(texture, str) and texture.strip():
        texture_name = texture.strip()
    else:
        texture_name = None

    return MaterialDefinition(
        name=name,
        color=color,
        texture=texture_name,
        double_sided=bool(data.get("double_sided", False)),
    )


def _parse_mesh(root: Path, data: Dict[str, Any]) -> Optional[MeshDefinition]:
    name = str(data.get("name", "")).strip()
    if not name:
        return None

    primitive = data.get("primitive") or data.get("type")
    path_value = data.get("path") or data.get("file")
    mesh_path: Optional[Path]
    if isinstance(path_value, str) and path_value.strip():
        mesh_path = Path(path_value)
        if not mesh_path.is_absolute():
            mesh_path = (root / mesh_path).resolve()
    else:
        mesh_path = None

    size = _coerce_vector(data.get("size"), 2, default=(1.0, 1.0))
    scale = _coerce_vector_optional(data.get("scale"), 3)
    offset = _coerce_vector(data.get("offset"), 3, default=(0.0, 0.0, 0.0))

    return MeshDefinition(
        name=name,
        path=mesh_path,
        primitive=str(primitive).strip().lower() if isinstance(primitive, str) else None,
        material=_optional_str(data.get("material")),
        size=size,
        up_axis=str(data.get("up_axis", "y")).strip().lower(),
        scale=scale,
        offset=offset,
    )


def _parse_scene_node(data: Dict[str, Any]) -> Optional[SceneNode]:
    mesh = _optional_str(data.get("mesh"))
    if mesh is None:
        return None
    name = _optional_str(data.get("name")) or mesh
    transform = _parse_transform(data.get("transform"))
    tags = ()
    raw_tags = data.get("tags")
    if isinstance(raw_tags, (list, tuple)):
        tags = tuple(str(tag) for tag in raw_tags if isinstance(tag, (str, int, float)))
    animation = data.get("animation") if isinstance(data.get("animation"), dict) else None
    return SceneNode(name=name, mesh=mesh, transform=transform, tags=tags, animation=animation)


def _parse_transform(data: Any) -> TransformDefinition:
    if not isinstance(data, dict):
        return TransformDefinition()
    translate = _coerce_vector(data.get("translate"), 3, default=(0.0, 0.0, 0.0))
    rotate = _coerce_vector(data.get("rotate_deg", data.get("rotation")), 3, default=(0.0, 0.0, 0.0))
    scale = _coerce_vector(data.get("scale"), 3, default=(1.0, 1.0, 1.0))
    return TransformDefinition(translate=translate, rotate_deg=rotate, scale=scale)


def _optional_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def _coerce_vector(value: Any, length: int, *, default: Tuple[float, ...]) -> Tuple[float, ...]:
    if isinstance(value, (list, tuple)) and len(value) >= length:
        try:
            return tuple(float(value[i]) for i in range(length))
        except (TypeError, ValueError):
            return default
    return default


def _coerce_vector_optional(value: Any, length: int) -> Optional[Tuple[float, ...]]:
    if isinstance(value, (list, tuple)) and len(value) >= length:
        try:
            return tuple(float(value[i]) for i in range(length))
        except (TypeError, ValueError):
            return None
    return None


def _resolve_wrap(value: Optional[str]) -> int:
    if not value:
        return GL_CLAMP_TO_EDGE
    normalized = value.strip().lower()
    if normalized in {"repeat"}:
        return GL_REPEAT
    if normalized in {"linear", "clamp", "clamp_to_edge"}:
        return GL_CLAMP_TO_EDGE
    return GL_CLAMP_TO_EDGE


def _resolve_filter(value: Optional[str]) -> int:
    if not value:
        return GL_LINEAR
    normalized = value.strip().lower()
    if normalized in {"nearest"}:
        return GL_NEAREST
    if normalized in {"linear_mipmap_linear", "trilinear"}:
        return GL_LINEAR_MIPMAP_LINEAR
    return GL_LINEAR


def _load_texture_pixels(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unsupported texture channel count: {image.shape[2]}")

    return np.ascontiguousarray(image, dtype=np.uint8)


def _build_geometry(definition: MeshDefinition) -> MeshGeometry:
    if definition.primitive == "plane":
        return _generate_plane(definition.size, definition.up_axis)

    if definition.path is None:
        raise HeadlessContextError(f"Mesh '{definition.name}' is missing a path/primitive")

    return _load_obj(definition.path)


def _generate_plane(size: Tuple[float, float], up_axis: str) -> MeshGeometry:
    half_w = float(size[0]) * 0.5
    half_d = float(size[1]) * 0.5

    if up_axis == "z":
        vertices = np.array(
            [
                [-half_w, -half_d, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [half_w, -half_d, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [half_w, half_d, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [-half_w, half_d, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:  # default to Y-up plane on the XZ axes
        vertices = np.array(
            [
                [-half_w, 0.0, -half_d, 0.0, 1.0, 0.0, 0.0, 0.0],
                [half_w, 0.0, -half_d, 0.0, 1.0, 0.0, 1.0, 0.0],
                [half_w, 0.0, half_d, 0.0, 1.0, 0.0, 1.0, 1.0],
                [-half_w, 0.0, half_d, 0.0, 1.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    bounds_min = vertices[:, 0:3].min(axis=0)
    bounds_max = vertices[:, 0:3].max(axis=0)
    return MeshGeometry(vertices=vertices, indices=indices, bounds_min=bounds_min, bounds_max=bounds_max)


def _load_obj(path: Path) -> MeshGeometry:
    positions: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []
    normals: List[Tuple[float, float, float]] = []
    faces: List[List[Tuple[int, Optional[int], Optional[int]]]] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            prefix, values = parts[0], parts[1:]
            if prefix == "v" and len(values) >= 3:
                positions.append((float(values[0]), float(values[1]), float(values[2])))
            elif prefix == "vt" and len(values) >= 2:
                texcoords.append((float(values[0]), float(values[1])))
            elif prefix == "vn" and len(values) >= 3:
                normals.append((float(values[0]), float(values[1]), float(values[2])))
            elif prefix == "f" and len(values) >= 3:
                face: List[Tuple[int, Optional[int], Optional[int]]] = []
                for token in values:
                    indices = token.split("/")
                    v_idx = _resolve_obj_index(indices[0], len(positions))
                    vt_idx = _resolve_obj_index(indices[1], len(texcoords)) if len(indices) > 1 else None
                    vn_idx = _resolve_obj_index(indices[2], len(normals)) if len(indices) > 2 else None
                    if v_idx is None:
                        continue
                    face.append((v_idx, vt_idx, vn_idx))
                if len(face) >= 3:
                    faces.append(face)

    if not faces:
        raise HeadlessContextError(f"OBJ '{path}' does not contain any faces")

    position_array = np.asarray(positions, dtype=np.float32)
    texcoord_array = np.asarray(texcoords, dtype=np.float32) if texcoords else None
    normal_array = np.asarray(normals, dtype=np.float32) if normals else None

    vertex_map: Dict[Tuple[int, Optional[int], Optional[int]], int] = {}
    vertices: List[np.ndarray] = []
    indices: List[int] = []

    for face in faces:
        for tri in _triangulate(face):
            p0 = position_array[tri[0][0]]
            p1 = position_array[tri[1][0]]
            p2 = position_array[tri[2][0]]
            face_normal = _compute_face_normal(p0, p1, p2)
            for corner in tri:
                key = (corner[0], corner[1], corner[2])
                mapped = vertex_map.get(key)
                if mapped is None:
                    position = position_array[corner[0]]
                    if corner[2] is not None and normal_array is not None:
                        normal = normal_array[corner[2]]
                    else:
                        normal = face_normal
                    if corner[1] is not None and texcoord_array is not None:
                        uv = texcoord_array[corner[1]]
                    else:
                        uv = np.array((0.0, 0.0), dtype=np.float32)
                    vertex = np.array(
                        [
                            position[0],
                            position[1],
                            position[2],
                            normal[0],
                            normal[1],
                            normal[2],
                            uv[0],
                            uv[1],
                        ],
                        dtype=np.float32,
                    )
                    mapped = len(vertices)
                    vertices.append(vertex)
                    vertex_map[key] = mapped
                indices.append(mapped)

    vertex_array = np.vstack(vertices) if vertices else np.zeros((0, 8), dtype=np.float32)
    index_array = np.asarray(indices, dtype=np.uint32)

    bounds_min = vertex_array[:, 0:3].min(axis=0) if vertex_array.size else np.zeros(3, dtype=np.float32)
    bounds_max = vertex_array[:, 0:3].max(axis=0) if vertex_array.size else np.zeros(3, dtype=np.float32)

    return MeshGeometry(
        vertices=vertex_array,
        indices=index_array,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def _resolve_obj_index(value: Optional[str], length: int) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed > 0:
        return parsed - 1
    if parsed < 0:
        return length + parsed
    return None


def _triangulate(face: Sequence[Tuple[int, Optional[int], Optional[int]]]) -> Iterable[Tuple[Tuple[int, Optional[int], Optional[int]], ...]]:
    if len(face) == 3:
        yield (face[0], face[1], face[2])
        return
    for idx in range(1, len(face) - 1):
        yield (face[0], face[idx], face[idx + 1])


def _compute_face_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    u = p1 - p0
    v = p2 - p0
    normal = np.cross(u, v)
    length = np.linalg.norm(normal)
    if length < 1e-8:
        return np.array((0.0, 1.0, 0.0), dtype=np.float32)
    return (normal / length).astype(np.float32)
