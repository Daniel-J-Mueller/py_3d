import json

from py_3d import EnvironmentDetailCluster, Mesh, ProceduralEnvironmentConfig, ProceduralEnvironmentGenerator, build_environment_chunk, ensure_procedural_world_assets, swayed_tree_primitives


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
    assert sum(len(chunk.grass.triangles) for chunk in chunks if chunk.grass is not None) > 0


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


def test_procedural_world_assets_save_with_world(tmp_path):
    config = ProceduralEnvironmentConfig(seed=5150, chunk_resolution=4, grass_blades_per_chunk=12, leaf_tufts_per_branch=1)

    manifest_path = ensure_procedural_world_assets(tmp_path / "world-assets", config)
    first_manifest = manifest_path.read_text(encoding="utf-8")
    second_manifest_path = ensure_procedural_world_assets(tmp_path / "world-assets", config)

    manifest = json.loads(first_manifest)
    assert second_manifest_path == manifest_path
    assert second_manifest_path.read_text(encoding="utf-8") == first_manifest
    assert manifest["format"] == "py_3d.procedural_world_assets.v1"
    assert manifest["seed"] == 5150
    assert (manifest_path.parent / "grass-tufts.json").exists()
    assert (manifest_path.parent / "maple-leaf-tufts.json").exists()


def test_distant_tree_lod_uses_layered_cards_not_canopy_blobs():
    config = ProceduralEnvironmentConfig(
        seed=909,
        chunk_resolution=4,
        tree_slots_per_axis=3,
        tree_density=1.0,
        bush_density=0.0,
        rock_density=0.0,
        grass_blades_per_chunk=0,
    )
    chunk = ProceduralEnvironmentGenerator(config).chunk((0, 0))
    meshes = [obj for obj in chunk.distant_objects if isinstance(obj, Mesh)]
    assert chunk.trees

    card_mesh = next(mesh for mesh in meshes[1:] if _max_repeated_triangle_start(mesh) >= 8)

    assert all(triangle.has_vertex_normals() for triangle in card_mesh.triangles[:12])


def test_tree_wind_sway_is_deterministic_and_gentle():
    config = ProceduralEnvironmentConfig(
        seed=910,
        chunk_resolution=4,
        tree_slots_per_axis=2,
        tree_density=1.0,
        grass_blades_per_chunk=0,
    )
    tree = ProceduralEnvironmentGenerator(config).chunk((0, 0)).trees[0]

    first = _object_vertex_signature(swayed_tree_primitives(tree, config.seed, wind_time=1.25))
    repeat = _object_vertex_signature(swayed_tree_primitives(tree, config.seed, wind_time=1.25))
    later = _object_vertex_signature(swayed_tree_primitives(tree, config.seed, wind_time=2.0))

    assert first == repeat
    assert first != later


def test_chunks_expose_bounds_biome_and_virtualized_tree_clusters():
    config = ProceduralEnvironmentConfig(
        seed=911,
        chunk_resolution=4,
        tree_slots_per_axis=3,
        tree_density=1.0,
        grass_blades_per_chunk=0,
    )
    chunk = ProceduralEnvironmentGenerator(config).chunk((0, 0))

    assert chunk.bounds_radius > config.chunk_size * 0.5
    assert chunk.bounds_min.x == 0.0
    assert chunk.bounds_max.z == config.chunk_size
    assert chunk.biome_name in {"wetland", "ridge", "forest", "meadow", "woodland"}
    assert chunk.tree_clusters
    assert all(isinstance(cluster, EnvironmentDetailCluster) for cluster in chunk.tree_clusters)
    assert all(cluster.objects and cluster.triangle_count > 0 for cluster in chunk.tree_clusters)
    assert {cluster.source_index for cluster in chunk.tree_clusters} <= set(range(len(chunk.trees)))


def test_hlod_proxy_is_compact_and_deterministic():
    config = ProceduralEnvironmentConfig(
        seed=912,
        chunk_resolution=8,
        tree_slots_per_axis=4,
        tree_density=1.0,
        bush_density=1.0,
        rock_density=1.0,
        grass_blades_per_chunk=20,
    )

    first = ProceduralEnvironmentGenerator(config).chunk((1, -1))
    second = ProceduralEnvironmentGenerator(config).chunk((1, -1))

    assert first.hlod_objects
    assert len(first.hlod_objects) < len(first.distant_objects)
    assert _object_vertex_signature(first.hlod_objects) == _object_vertex_signature(second.hlod_objects)
    assert _triangle_count(first.hlod_objects) < _triangle_count(first.distant_objects)


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
            round(triangle.a.x, 4),
            round(triangle.a.y, 4),
            round(triangle.a.z, 4),
            round(triangle.b.x, 4),
            round(triangle.b.y, 4),
            round(triangle.b.z, 4),
            round(triangle.c.x, 4),
            round(triangle.c.y, 4),
            round(triangle.c.z, 4),
            triangle.material.color.to_tuple(),
        )
        for triangle in (chunk.grass.triangles if chunk.grass is not None else ())[:24]
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
    distant_count = len(chunk.distant_objects)
    return terrain_vertices, water_vertices, trees, bushes, rocks, grass, shoreline_count, ripple_count, sources, distant_count


def _object_vertex_signature(objects):
    values = []
    for obj in objects:
        if not isinstance(obj, Mesh):
            continue
        for triangle in obj.triangles[:12]:
            values.extend(
                (
                    round(vertex.x, 4),
                    round(vertex.y, 4),
                    round(vertex.z, 4),
                )
                for vertex in (triangle.a, triangle.b, triangle.c)
            )
    return tuple(values)


def _triangle_count(objects):
    total = 0
    for obj in objects:
        if isinstance(obj, Mesh):
            total += len(obj.triangles)
    return total


def _max_repeated_triangle_start(mesh):
    fan_centers: dict[tuple[float, float, float], int] = {}
    for triangle in mesh.triangles:
        key = (round(triangle.a.x, 3), round(triangle.a.y, 3), round(triangle.a.z, 3))
        fan_centers[key] = fan_centers.get(key, 0) + 1
    return max(fan_centers.values()) if fan_centers else 0


def _edge_vertices(mesh, *, x: float):
    values = {
        (round(vertex.z, 5), round(vertex.y, 5))
        for triangle in mesh.triangles
        for vertex in (triangle.a, triangle.b, triangle.c)
        if round(vertex.x, 5) == round(x, 5)
    }
    return tuple(sorted(values))
