from py_3d import Camera


def test_camera_projects_target_to_screen_center():
    camera = Camera(position=(0, 0, -5), target=(0, 0, 0))

    point = camera.project((0, 0, 0), width=101, height=101)

    assert point is not None
    assert point.x == 50
    assert point.y == 50
    assert point.depth == 5


def test_camera_rejects_points_behind_near_plane():
    camera = Camera(position=(0, 0, -5), target=(0, 0, 0), near=0.5)

    assert camera.project((0, 0, -5), width=100, height=100) is None


def test_camera_projects_world_right_to_screen_right():
    camera = Camera(position=(0, 0, -5), target=(0, 0, 0))

    center = camera.project((0, 0, 0), width=101, height=101)
    right = camera.project((1, 0, 0), width=101, height=101)

    assert center is not None
    assert right is not None
    assert right.x > center.x


def test_first_person_camera_uses_subject_eye_height_and_forward():
    camera = Camera.first_person((1, 0, 2), (0, 0, 1), eye_height=1.6)

    assert camera.position.y == 1.6
    assert camera.target.z > camera.position.z


def test_third_person_camera_sits_behind_subject():
    camera = Camera.third_person((0, 0, 0), (0, 0, 1), distance=3.0, height=1.2)

    assert camera.position.z < 0.0
    assert camera.position.y == 1.2
