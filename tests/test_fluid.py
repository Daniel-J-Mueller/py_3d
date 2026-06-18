from py_3d import FluidBlob, FluidWorld, Vec3


def test_fluid_blob_radius_preserves_volume():
    blob = FluidBlob.from_radius((0, 0, 0), 0.5)

    assert blob.radius == 0.5


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
