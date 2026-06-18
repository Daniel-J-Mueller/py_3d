"""Primitive 3D pixel drawing and rendering foundations."""

from .buffer import DepthBuffer, PixelBuffer
from .camera import Camera, ProjectedPoint
from .collision import BowlCollider, BoxCollider, CompoundSphereCollider, PlaneCollider, SphereCollider
from .color import Color
from .fluid import FluidBlob, FluidWorld
from . import draw
from .gpu import GPURenderer, GPUSceneBatch, build_gpu_scene_batch, detect_gpu_backends
from .importers import load_obj, load_stl
from .lights import Lamp, LightSample, Sun
from .materials import Material
from .math3d import Vec3, as_vec3, clamp
from .noise import FractalNoise3D, SurfacePerturbation, ValueNoise3D
from .overlays import TextBulletin
from .physics import KinematicBowl, PhysicsWorld, SphereBody, StaticBox, StaticPlane, World
from .primitives import Bowl, Box, Capsule, Line3, Mesh, Plane, Point3, Sphere, Triangle
from .render import CPURenderer, RenderEngine, RenderSettings, Renderer
from .scene import Scene
from .textures import planar_project_triangles

__all__ = [
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
    "GPURenderer",
    "GPUSceneBatch",
    "Lamp",
    "LightSample",
    "Line3",
    "KinematicBowl",
    "Material",
    "Mesh",
    "PixelBuffer",
    "Plane",
    "PlaneCollider",
    "Point3",
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
    "World",
    "as_vec3",
    "build_gpu_scene_batch",
    "clamp",
    "draw",
    "detect_gpu_backends",
    "load_obj",
    "load_stl",
    "planar_project_triangles",
]
