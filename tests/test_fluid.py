import importlib.util
from math import cos, sin, tau

import pytest

from py_3d import BlobSurface, FluidBlob, FluidWorld, GPUVectorFluidWorld, Vec3, VectorFluidParticle, VectorFluidWorld


def test_fluid_blob_radius_preserves_volume():
    blob = FluidBlob.from_radius((0, 0, 0), 0.5)

    assert blob.radius == 0.5
    assert isinstance(blob.to_primitive(), BlobSurface)


def test_fluid_blob_surface_deforms_from_stretch():
    relaxed = FluidBlob.from_radius((0, 0, 0), 0.5)
    stretched = FluidBlob.from_radius((0, 0, 0), 0.5, stretch=(1.2, 0, 0), wetting=0.4, stickiness=0.3)

    assert relaxed.to_primitive().to_triangles(segments=8, rings=4) != stretched.to_primitive().to_triangles(segments=8, rings=4)
    assert stretched.wetting == 0.4
    assert stretched.stickiness == 0.3


def test_fluid_world_splits_overstretched_blob_and_preserves_volume():
    world = FluidWorld(gravity=(0, 0, 0), heal_distance_factor=0.0)
    blob = FluidBlob.from_radius((0, 1, 0), 0.35, stretch=(2.0, 0, 0), stretchiness=0.0)
    world.add_blob(blob)
    before = world.total_volume()

    world.step(0.1)

    assert len(world.blobs) == 2
    assert abs(world.total_volume() - before) < 1e-9


def test_fluid_world_heals_close_blobs():
    world = FluidWorld(gravity=(0, 0, 0), heal_distance_factor=2.0)
    first = FluidBlob.from_radius((0, 1, 0), 0.25)
    second = FluidBlob.from_radius((0.1, 1, 0), 0.25)
    world.add_blob(first)
    world.add_blob(second)

    world.step(0.01)

    assert len(world.blobs) == 1
    assert isinstance(world.blobs[0].position, Vec3)


def test_vector_fluid_repels_close_particles_and_accepts_external_force():
    world = VectorFluidWorld(bounds_min=(-2, -2, -2), bounds_max=(2, 2, 2), gravity=(0, 0, 0), rest_distance=0.5, repel_strength=4.0, self_attraction=0.0)
    first = world.add_particle(VectorFluidParticle((0.0, 0.0, 0.0)))
    second = world.add_particle(VectorFluidParticle((0.1, 0.0, 0.0)))

    world.step(0.1, external_force=lambda particle, dt: Vec3(0.0, 1.0, 0.0))

    assert first.position.x < 0.0
    assert second.position.x > 0.1
    assert first.velocity.y > 0.0


def test_vector_fluid_accepts_tuple_external_force_without_vec3_allocation():
    world = VectorFluidWorld(bounds_min=(-2, -2, -2), bounds_max=(2, 2, 2), gravity=(0, 0, 0), rest_distance=0.5, repel_strength=0.0, self_attraction=0.0)
    particle = world.add_particle(VectorFluidParticle((0.0, 0.0, 0.0)))

    world.step(0.1, external_force=lambda _particle, _dt: (0.0, 2.0, 0.0))

    assert particle.velocity.y > 0.0


def test_vector_fluid_dampens_close_particle_vibration():
    world = VectorFluidWorld(
        bounds_min=(-2, -2, -2),
        bounds_max=(2, 2, 2),
        gravity=(0, 0, 0),
        rest_distance=0.5,
        repel_strength=0.0,
        self_attraction=0.0,
        close_damping=10.0,
    )
    first = world.add_particle(VectorFluidParticle((0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0)))
    second = world.add_particle(VectorFluidParticle((0.1, 0.0, 0.0), velocity=(-1.0, 0.0, 0.0)))

    world.step(0.1)

    assert abs(first.velocity.x - second.velocity.x) < 2.0


def test_vector_fluid_gas_has_no_gravity_or_self_attraction_by_default():
    world = VectorFluidWorld.gas(bounds_min=(-2, -2, -2), bounds_max=(2, 2, 2))

    assert world.gravity == Vec3(0.0, 0.0, 0.0)
    assert world.self_attraction == 0.0


def test_gpu_vector_fluid_steps_particles_in_compute_context():
    if importlib.util.find_spec("glfw") is None or importlib.util.find_spec("moderngl") is None:
        pytest.skip("glfw/moderngl are not installed")

    import glfw
    import moderngl

    if not glfw.init():
        pytest.skip("GLFW could not initialize")
    window = None
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        window = glfw.create_window(16, 16, "gpu-fluid-test", None, None)
        if not window:
            pytest.skip("OpenGL 4.3 context unavailable")
        glfw.make_context_current(window)
        ctx = moderngl.create_context()
        if ctx.version_code < 430:
            pytest.skip("OpenGL compute shaders unavailable")
        world = GPUVectorFluidWorld.liquid(
            ctx,
            bounds_min=(-2, -2, -2),
            bounds_max=(2, 2, 2),
            gravity=(0, 0, 0),
            rest_distance=0.25,
            repel_strength=2.0,
            self_attraction=0.4,
        )
        for index in range(16):
            angle = tau * index / 16
            world.add_particle(VectorFluidParticle((cos(angle) * 0.16, 0.0, sin(angle) * 0.16)))
        before = [particle.position for particle in world.particles]

        world.step(0.02)

        assert any(particle.position.distance_to(start) > 1e-5 for particle, start in zip(world.particles, before))
        world.close()
    finally:
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def test_gpu_vector_fluid_spatial_grid_neighbor_path():
    if importlib.util.find_spec("glfw") is None or importlib.util.find_spec("moderngl") is None:
        pytest.skip("glfw/moderngl are not installed")

    import glfw
    import moderngl

    if not glfw.init():
        pytest.skip("GLFW could not initialize")
    window = None
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        window = glfw.create_window(16, 16, "gpu-fluid-grid-test", None, None)
        if not window:
            pytest.skip("OpenGL 4.3 context unavailable")
        glfw.make_context_current(window)
        ctx = moderngl.create_context()
        if ctx.version_code < 430:
            pytest.skip("OpenGL compute shaders unavailable")
        world = GPUVectorFluidWorld.liquid(
            ctx,
            bounds_min=(-1, -1, -1),
            bounds_max=(1, 1, 1),
            gravity=(0, 0, 0),
            rest_distance=0.18,
            repel_strength=2.0,
            self_attraction=0.4,
        )
        world.spatial_grid_min_particles = 1
        for index in range(32):
            angle = tau * index / 32
            world.add_particle(VectorFluidParticle((cos(angle) * 0.16, 0.0, sin(angle) * 0.16)))

        world.step(0.02)

        assert world.spatial_grid_active is True
        assert world._grid_signature is not None
        world.close()
    finally:
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
