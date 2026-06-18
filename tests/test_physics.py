from py_3d import PhysicsWorld, SphereBody, StaticBox, StaticPlane, Vec3


def test_sphere_collides_with_static_plane():
    body = SphereBody(position=(0, 0.1, 0), radius=0.5, velocity=(0, -1, 0), restitution=0.5)
    plane = StaticPlane(point=(0, 0, 0), normal=(0, 1, 0), restitution=1.0)
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_sphere(body)
    world.add_plane(plane)

    world.step(0.1)

    assert body.position.y >= 0.5
    assert body.velocity.y > 0


def test_sphere_collides_with_static_box_wall():
    body = SphereBody(position=(-0.8, 0, 0), radius=0.4, velocity=(2, 0, 0), restitution=1.0, friction=0.0)
    wall = StaticBox(center=(0, 0, 0), size=(0.4, 2, 2), restitution=1.0, friction=0.0)
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_sphere(body)
    world.add_box(wall)

    world.step(0.25)

    assert body.position.x <= -0.6
    assert body.velocity.x < 0


def test_world_steps_tilted_plane_motion():
    body = SphereBody(position=(-1, 0.8, 0), radius=0.2, friction=0.0)
    ramp = StaticPlane(point=(0, 0, 0), normal=(0.4, 1, 0), friction=0.0)
    world = PhysicsWorld(gravity=(0, -9.81, 0))
    world.add_sphere(body)
    world.add_plane(ramp)
    start = body.position

    for _ in range(20):
        world.step(1 / 60, substeps=2)

    assert isinstance(body.position, Vec3)
    assert body.position.x > start.x
