"""Primitive 3D pixel drawing and rendering foundations."""

from .buffer import DepthBuffer, PixelBuffer
from .camera import Camera, ProjectedPoint
from .assets import MESH_ASSET_FORMAT, load_mesh_asset, mesh_asset_metadata
from .collision import BowlCollider, BoxCollider, CompoundSphereCollider, PlaneCollider, SphereCollider
from .color import Color
from .fluid import FluidBlob, FluidWorld, GPUVectorFluidWorld, VectorFluidParticle, VectorFluidWorld
from . import draw
from .environment import BushInstance, EnvironmentChunk, ProceduralEnvironmentConfig, ProceduralEnvironmentGenerator, RockInstance, TreeInstance, WaterSource, build_environment_chunk, ensure_procedural_world_assets, swayed_tree_primitives
from .gpu import GPURenderer, GPUSceneBatch, build_gpu_scene_batch, detect_gpu_backends
from .hud import HUDAnimation, HUDImage, HUDRect, HUDText, LiveHUD
from .importers import load_obj, load_stl
from .inventory import CubePlacer, Inventory, InventorySlot, place_cube_from_inventory
from .lights import Lamp, LightSample, Sun
from .live_defaults import CANONICAL_CAMERA_MODES, CANONICAL_LIVE_MENU_ACTIONS, canonical_live_menu_options, canonical_player_movement_key, next_camera_mode, update_canonical_live_menu
from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .noise import FractalNoise3D, SurfacePerturbation, ValueNoise3D
from .overlays import FloatingTextBulletin, TextBulletin
from .player import PlayerModel
from .physics import KinematicBowl, PhysicsWorld, SphereBody, StaticBox, StaticPlane, World
from .portal import PortalPair, PortalSurface, portal_camera_for, scene_with_portal_textures
from .primitives import BlobSurface, Bowl, Box, Capsule, HangingConeLampPrimitive, LampPrimitive, Line3, LineSet3, Mesh, ParticleFluidPuddle, ParticleFluidVolume, ParticleWaterSurface, Plane, Point3, Sphere, TransformedMesh, Triangle
from .render import CPURenderer, RenderEngine, RenderSettings, Renderer
from .scene import Scene
from .sky import SkyPrefab
from .textures import planar_project_triangles
from .window import PixelWindow, WindowEvent

__all__ = [
    "BlobSurface",
    "BushInstance",
    "Box",
    "BoxCollider",
    "Bowl",
    "BowlCollider",
    "CompoundSphereCollider",
    "CPURenderer",
    "Camera",
    "Capsule",
    "Color",
    "CANONICAL_CAMERA_MODES",
    "CANONICAL_LIVE_MENU_ACTIONS",
    "CubePlacer",
    "DepthBuffer",
    "EnvironmentChunk",
    "FluidBlob",
    "FluidWorld",
    "FloatingTextBulletin",
    "GPURenderer",
    "GPUSceneBatch",
    "GPUVectorFluidWorld",
    "HangingConeLampPrimitive",
    "HUDAnimation",
    "HUDImage",
    "HUDRect",
    "HUDText",
    "Inventory",
    "InventorySlot",
    "Lamp",
    "LampPrimitive",
    "LightSample",
    "LiveHUD",
    "Line3",
    "LineSet3",
    "MESH_ASSET_FORMAT",
    "KinematicBowl",
    "Material",
    "Mesh",
    "ParticleFluidPuddle",
    "ParticleFluidVolume",
    "ParticleWaterSurface",
    "PixelBuffer",
    "Plane",
    "PlaneCollider",
    "Point3",
    "PlayerModel",
    "ProceduralEnvironmentConfig",
    "ProceduralEnvironmentGenerator",
    "PortalPair",
    "PortalSurface",
    "ProjectedPoint",
    "PhysicsWorld",
    "RenderEngine",
    "RenderSettings",
    "Renderer",
    "RockInstance",
    "Scene",
    "Sphere",
    "SphereCollider",
    "SphereBody",
    "StaticBox",
    "StaticPlane",
    "FractalNoise3D",
    "SkyPrefab",
    "SurfacePerturbation",
    "Sun",
    "TextBulletin",
    "Triangle",
    "TreeInstance",
    "TransformedMesh",
    "PixelWindow",
    "ValueNoise3D",
    "Vec3",
    "VectorFluidParticle",
    "VectorFluidWorld",
    "WindowEvent",
    "World",
    "WaterSource",
    "as_vec3",
    "build_gpu_scene_batch",
    "build_environment_chunk",
    "clamp",
    "canonical_live_menu_options",
    "canonical_player_movement_key",
    "draw",
    "detect_gpu_backends",
    "ensure_procedural_world_assets",
    "load_mesh_asset",
    "load_obj",
    "load_stl",
    "mesh_asset_metadata",
    "next_camera_mode",
    "planar_project_triangles",
    "place_cube_from_inventory",
    "portal_camera_for",
    "scene_with_portal_textures",
    "swayed_tree_primitives",
    "update_canonical_live_menu",
]
