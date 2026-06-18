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
from .lights import Lamp, LightSample, Sun
from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .noise import FractalNoise3D, SurfacePerturbation, ValueNoise3D
from .overlays import FloatingTextBulletin, TextBulletin
from .player import PlayerModel
from .physics import KinematicBowl, PhysicsWorld, SphereBody, StaticBox, StaticPlane, World
from .primitives import BlobSurface, Bowl, Box, Capsule, HangingConeLampPrimitive, LampPrimitive, Line3, Mesh, Plane, Point3, Sphere, Triangle
from .render import CPURenderer, RenderEngine, RenderSettings, Renderer
from .scene import Scene
from .textures import planar_project_triangles

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
    "SurfacePerturbation",
    "Sun",
    "TextBulletin",
    "Triangle",
    "ValueNoise3D",
    "Vec3",
    "VectorFluidParticle",
    "VectorFluidWorld",
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
]
