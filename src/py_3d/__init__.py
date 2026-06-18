"""Primitive 3D pixel drawing and rendering foundations."""

from .buffer import DepthBuffer, PixelBuffer
from .camera import Camera, ProjectedPoint
from .assets import MESH_ASSET_FORMAT, load_mesh_asset, mesh_asset_metadata
from .collision import BowlCollider, BoxCollider, CompoundSphereCollider, PlaneCollider, SphereCollider
from .color import Color
from .fluid import FluidBlob, FluidWorld, VectorFluidParticle, VectorFluidWorld
from . import draw
from .gpu import GPURenderer, GPUSceneBatch, build_gpu_scene_batch, detect_gpu_backends
from .hud import HUDAnimation, HUDImage, HUDRect, HUDText, LiveHUD
from .importers import load_obj, load_stl
from .inventory import CubePlacer, Inventory, InventorySlot, place_cube_from_inventory
from .lights import Lamp, LightSample, Sun
from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .noise import FractalNoise3D, SurfacePerturbation, ValueNoise3D
from .overlays import FloatingTextBulletin, TextBulletin
from .player import PlayerModel
from .physics import KinematicBowl, PhysicsWorld, SphereBody, StaticBox, StaticPlane, World
from .portal import PortalPair, PortalSurface, portal_camera_for, scene_with_portal_textures
from .primitives import BlobSurface, Bowl, Box, Capsule, HangingConeLampPrimitive, LampPrimitive, Line3, Mesh, Plane, Point3, Sphere, Triangle
from .render import CPURenderer, RenderEngine, RenderSettings, Renderer
from .scene import Scene
from .sky import SkyPrefab
from .textures import planar_project_triangles
from .window import PixelWindow, WindowEvent

__all__ = [
    "BlobSurface",
    "Box",
    "BoxCollider",
    "Bowl",
    "BowlCollider",
    "CompoundSphereCollider",
    "CPURenderer",
    "Camera",
    "Capsule",
    "Color",
    "CubePlacer",
    "DepthBuffer",
    "FluidBlob",
    "FluidWorld",
    "FloatingTextBulletin",
    "GPURenderer",
    "GPUSceneBatch",
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
    "MESH_ASSET_FORMAT",
    "KinematicBowl",
    "Material",
    "Mesh",
    "PixelBuffer",
    "Plane",
    "PlaneCollider",
    "Point3",
    "PlayerModel",
    "PortalPair",
    "PortalSurface",
    "ProjectedPoint",
    "PhysicsWorld",
    "RenderEngine",
    "RenderSettings",
    "Renderer",
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
    "PixelWindow",
    "ValueNoise3D",
    "Vec3",
    "VectorFluidParticle",
    "VectorFluidWorld",
    "WindowEvent",
    "World",
    "as_vec3",
    "build_gpu_scene_batch",
    "clamp",
    "draw",
    "detect_gpu_backends",
    "load_mesh_asset",
    "load_obj",
    "load_stl",
    "mesh_asset_metadata",
    "planar_project_triangles",
    "place_cube_from_inventory",
    "portal_camera_for",
    "scene_with_portal_textures",
]
