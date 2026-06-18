from py_3d import BlobSurface, FluidBlob, FluidWorld, Vec3, VectorFluidParticle, VectorFluidWorld


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
