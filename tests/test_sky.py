from py_3d import RenderSettings, Scene, SkyPrefab, Sun


def test_sky_prefab_adds_sun_from_manual_angle():
    sky = SkyPrefab(manual_sun_angle=True, sun_azimuth_degrees=0.0, sun_elevation_degrees=45.0)
    scene = Scene()

    sky.apply(scene)

    suns = [light for light in scene.lights if isinstance(light, Sun)]
    assert len(suns) == 1
    assert suns[0].direction.y < 0.0
    assert scene.background != RenderSettings().background


def test_sky_cycle_advances_day_and_night_with_separate_lengths():
    sky = SkyPrefab(time_of_day=6.0, cycle_enabled=True, day_length_seconds=12.0, night_length_seconds=6.0)

    sky.step(6.0)
    noonish = sky.time_of_day
    sky.set_time(18.0)
    sky.step(3.0)

    assert 11.9 <= noonish <= 12.1
    assert 23.9 <= sky.time_of_day or sky.time_of_day <= 0.1


def test_sky_stars_and_clouds_are_toggleable():
    night = SkyPrefab(time_of_day=0.0, stars_enabled=True, clouds_enabled=True)
    day = SkyPrefab(time_of_day=12.0, stars_enabled=True, clouds_enabled=True)

    assert len(night.star_primitives()) > 0
    assert len(day.cloud_primitives()) > 0
    night.toggle_stars()
    day.toggle_clouds()

    assert not night.stars_enabled
    assert not day.clouds_enabled


def test_sky_settings_for_updates_background_only():
    sky = SkyPrefab(time_of_day=12.0)
    settings = RenderSettings(width=123, height=77, reflection_bounces=2)

    updated = sky.settings_for(settings)

    assert updated.width == 123
    assert updated.height == 77
    assert updated.reflection_bounces == 2
    assert updated.background == sky.background_color()
