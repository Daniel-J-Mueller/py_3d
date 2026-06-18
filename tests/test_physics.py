from py_3d import (
    Bowl,
    BowlCollider,
    BoxCollider,
    KinematicBowl,
    PhysicsWorld,
    PlaneCollider,
    SphereBody,
    SphereCollider,
    StaticBox,
    StaticPlane,
    SurfacePerturbation,
    Vec3,
)


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


def test_collision_boundary_defaults_to_render_sphere_radius():
    body = SphereBody(position=(0, 0.1, 0), radius=0.75, velocity=(0, -1, 0), restitution=0.0)
    plane = StaticPlane(point=(0, 0, 0), normal=(0, 1, 0), restitution=0.0)
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_sphere(body)
    world.add_plane(plane)

    world.step(0.1)

    assert body.position.y >= 0.75
    assert body.collision_boundary is None
    assert body.collision_radius() == 0.75


def test_collision_boundary_override_can_differ_from_render_geometry():
    body = SphereBody(
        position=(0, 0.1, 0),
        radius=1.0,
        velocity=(0, -1, 0),
        restitution=0.0,
        collision_boundary=SphereCollider(radius=0.25),
    )
    plane = StaticPlane(point=(0, 0, 0), normal=(0, 1, 0), restitution=0.0)
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_sphere(body)
    world.add_plane(plane)

    world.step(0.1)

    assert 0.25 <= body.position.y < 1.0
    assert body.to_primitive().radius == 1.0
    assert body.to_collision_primitive().radius == 0.25


def test_collision_boundary_sync_can_replace_override_from_render_geometry():
    body = SphereBody(position=(0, 0, 0), radius=0.5, collision_boundary=SphereCollider(radius=0.2))

    assert body.sync_collision_boundary().radius == 0.2
    body.radius = 0.8
    synced = body.sync_collision_boundary(force=True)

    assert synced.radius == 0.8
    assert body.collision_radius() == 0.8


def test_synced_collision_boundary_includes_surface_perturbation_margin():
    body = SphereBody(
        position=(0, 0, 0),
        radius=1.0,
        visual_perturbation=SurfacePerturbation(magnitude=0.2, scale=2.0),
    )

    assert body.collision_boundary is None
    assert body.collision_radius() == 1.2


def test_static_box_collision_boundary_override():
    body = SphereBody(position=(-0.8, 0, 0), radius=0.4, velocity=(2, 0, 0), restitution=1.0, friction=0.0)
    visual_wall = StaticBox(
        center=(0, 0, 0),
        size=(0.4, 2, 2),
        restitution=1.0,
        friction=0.0,
        collision_boundary=BoxCollider(size=(0.1, 2, 2), offset=(0.4, 0, 0)),
    )
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_sphere(body)
    world.add_box(visual_wall)

    world.step(0.25)

    assert body.position.x > -0.6
    assert body.velocity.x > 0


def test_static_plane_collision_boundary_override():
    body = SphereBody(position=(0, 0.1, 0), radius=0.3, velocity=(0, -1, 0), restitution=0.0)
    visual_floor_below_collision_floor = StaticPlane(
        point=(0, -1, 0),
        normal=(0, 1, 0),
        restitution=0.0,
        collision_boundary=PlaneCollider(point=(0, 0, 0), normal=(0, 1, 0)),
    )
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_sphere(body)
    world.add_plane(visual_floor_below_collision_floor)

    world.step(0.1)

    assert body.position.y >= 0.3


def test_bowl_primitive_generates_open_cap_triangles():
    bowl = Bowl(center=(0, 0, 0), radius=1.0, depth=1.0)

    triangles = bowl.to_triangles(segments=8, rings=4)

    assert len(triangles) == 56
    assert all(vertex.y <= 1e-12 for triangle in triangles for vertex in (triangle.a, triangle.b, triangle.c))


def test_kinematic_bowl_set_center_updates_velocity_and_syncs_collider():
    bowl = KinematicBowl(center=(0, 0, 0), radius=1.25, depth=0.75, collision_boundary=BowlCollider(radius=0.8))

    bowl.set_center((0, 0.5, 0), dt=0.25)
    synced = bowl.sync_collision_boundary(force=True)

    assert bowl.velocity == Vec3(0.0, 2.0, 0.0)
    assert synced.radius == 1.25
    assert synced.depth == 0.75


def test_sphere_collides_with_kinematic_bowl_wall():
    fruit = SphereBody(position=(1.05, -0.1, 0), radius=0.25, velocity=(1.0, 0, 0), restitution=1.0, friction=0.0)
    bowl = KinematicBowl(center=(0, 0, 0), radius=1.2, restitution=1.0, friction=0.0)
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_sphere(fruit)
    world.add_bowl(bowl)

    world.step(0.1)

    assert fruit.collision_center().distance_to(bowl.collision_center()) <= bowl.radius - fruit.collision_radius() + 1e-9
    assert fruit.velocity.x < 0.0


def test_spheres_collide_with_each_other():
    left = SphereBody(position=(-0.25, 0, 0), radius=0.3, velocity=(1.0, 0, 0), restitution=1.0, friction=0.0)
    right = SphereBody(position=(0.25, 0, 0), radius=0.3, velocity=(-1.0, 0, 0), restitution=1.0, friction=0.0)
    world = PhysicsWorld(gravity=(0, 0, 0))
    world.add_sphere(left)
    world.add_sphere(right)

    world.step(0.01)

    assert left.collision_center().distance_to(right.collision_center()) >= left.radius + right.radius
    assert left.velocity.x < 0.0
    assert right.velocity.x > 0.0
