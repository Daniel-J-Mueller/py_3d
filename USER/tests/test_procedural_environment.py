from py_3d import ProceduralEnvironmentConfig, ProceduralEnvironmentGenerator, build_environment_chunk


def test_procedural_chunks_regenerate_without_feature_changes():
    config = ProceduralEnvironmentConfig(seed=1234, chunk_resolution=4, tree_slots_per_axis=3, tree_density=0.45, grass_blades_per_chunk=24)
    generator = ProceduralEnvironmentGenerator(config)

    first = _chunk_signature(generator.chunk((2, -3)))
    generator.clear_cache()
    second = _chunk_signature(generator.chunk((2, -3)))

    assert first == second


def test_procedural_terrain_is_continuous_across_chunk_edges():
    config = ProceduralEnvironmentConfig(seed=99, chunk_size=12.0, chunk_resolution=6)
    generator = ProceduralEnvironmentGenerator(config)
    left = generator.chunk((0, 0))
    right = generator.chunk((1, 0))

    left_edge = _edge_vertices(left.terrain, x=config.chunk_size)
    right_edge = _edge_vertices(right.terrain, x=config.chunk_size)

    assert left_edge == right_edge


def test_procedural_scene_streams_water_and_foliage_near_origin():
    config = ProceduralEnvironmentConfig(chunk_resolution=6, grass_blades_per_chunk=36)
    generator = ProceduralEnvironmentGenerator(config)
    chunks = generator.chunks_around((0.0, 0.0, 0.0), radius=2)

    assert any(chunk.water is not None for chunk in chunks)
    assert sum(len(chunk.trees) for chunk in chunks) > 0
    assert sum(len(chunk.bushes) for chunk in chunks) > 0
    assert sum(len(chunk.rocks) for chunk in chunks) > 0
    assert sum(len(chunk.grass.segments) for chunk in chunks if chunk.grass is not None) > 0


def test_returning_to_evicted_area_keeps_environment_stable():
    config = ProceduralEnvironmentConfig(seed=2026, chunk_resolution=5, tree_slots_per_axis=4, tree_density=0.5, grass_blades_per_chunk=24)
    generator = ProceduralEnvironmentGenerator(config)

    home = _chunk_signature(generator.chunk((0, 0)))
    generator.chunks_around((128.0, 0.0, -96.0), radius=1)
    generator.prune_cache_around((128.0, 0.0, -96.0), radius=1, margin=0)
    assert generator.cached_chunk_count <= 9
    returned = _chunk_signature(generator.chunk((0, 0)))

    assert returned == home


def test_cached_chunk_queries_do_not_build_missing_chunks():
    config = ProceduralEnvironmentConfig(seed=44, chunk_resolution=4)
    generator = ProceduralEnvironmentGenerator(config)

    assert generator.cached_chunks_around((0.0, 0.0, 0.0), radius=1) == ()
    assert generator.cached_chunk((0, 0)) is None
    assert generator.cached_chunk_count == 0

    center = generator.chunk((0, 0))

    assert generator.cached_chunks_around((0.0, 0.0, 0.0), radius=1) == (center,)
    assert generator.cached_chunk_count == 1


def test_worker_chunk_builder_matches_local_generator():
    config = ProceduralEnvironmentConfig(seed=88, chunk_resolution=4, tree_slots_per_axis=3, tree_density=0.4, grass_blades_per_chunk=20)

    local = ProceduralEnvironmentGenerator(config).chunk((1, -2))
    worker = build_environment_chunk(config, (1, -2))

    assert _chunk_signature(worker) == _chunk_signature(local)


def _chunk_signature(chunk):
    terrain_vertices = tuple(
        tuple(round(value, 4) for value in vertex)
        for triangle in chunk.terrain.triangles[:12]
        for vertex in (triangle.a, triangle.b, triangle.c)
    )
    water_vertices = ()
    if chunk.water is not None:
        water_vertices = tuple(
            tuple(round(value, 4) for value in vertex)
            for triangle in chunk.water.triangles[:12]
            for vertex in (triangle.a, triangle.b, triangle.c)
        )
    trees = tuple(
        (
            round(tree.position.x, 4),
            round(tree.position.y, 4),
            round(tree.position.z, 4),
            round(tree.trunk_height, 4),
            round(tree.trunk_radius, 4),
            round(tree.crown_radius, 4),
            round(tree.yaw, 4),
            round(tree.lean.x, 4),
            round(tree.lean.z, 4),
            tree.branch_count,
            tree.species,
            tree.material_variant,
        )
        for tree in chunk.trees
    )
    bushes = tuple(
        (
            round(bush.position.x, 4),
            round(bush.position.y, 4),
            round(bush.position.z, 4),
            round(bush.radius, 4),
            round(bush.height, 4),
            bush.variant,
        )
        for bush in chunk.bushes
    )
    rocks = tuple(
        (
            round(rock.position.x, 4),
            round(rock.position.y, 4),
            round(rock.position.z, 4),
            round(rock.radius, 4),
            round(rock.height, 4),
            round(rock.yaw, 4),
            rock.variant,
        )
        for rock in chunk.rocks
    )
    grass = tuple(
        (
            round(start.x, 4),
            round(start.y, 4),
            round(start.z, 4),
            round(end.x, 4),
            round(end.y, 4),
            round(end.z, 4),
        )
        for start, end in (chunk.grass.segments if chunk.grass is not None else ())[:24]
    )
    sources = tuple(
        (
            round(source.center.x, 4),
            round(source.center.y, 4),
            round(source.center.z, 4),
            round(source.radius, 4),
            round(source.level, 4),
        )
        for source in chunk.water_sources
    )
    shoreline_count = len(chunk.shorelines.segments) if chunk.shorelines is not None else 0
    ripple_count = len(chunk.ripples.segments) if chunk.ripples is not None else 0
    return terrain_vertices, water_vertices, trees, bushes, rocks, grass, shoreline_count, ripple_count, sources


def _edge_vertices(mesh, *, x: float):
    values = {
        (round(vertex.z, 5), round(vertex.y, 5))
        for triangle in mesh.triangles
        for vertex in (triangle.a, triangle.b, triangle.c)
        if round(vertex.x, 5) == round(x, 5)
    }
    return tuple(sorted(values))
