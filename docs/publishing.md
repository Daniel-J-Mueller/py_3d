# Publishing

This repository publishes as `py3dengine` and imports as `py_3d`.

The PyPI distribution name in `pyproject.toml` is `py3dengine`. Users install
that distribution name with `pip`, then import the Python package as `py_3d`.
The names do not need to match.

Trusted Publishing is configured on PyPI with:

| PyPI field | Value |
| --- | --- |
| PyPI Project Name | `py3dengine` |
| Owner | `Daniel-J-Mueller` |
| Repository name | `py_3d` |
| Workflow name | `publish.yml` |
| Environment name | any / blank |

Release flow:

1. Update `version` in `pyproject.toml`.
2. Commit and push the release changes.
3. Create and push a matching release tag:

   ```powershell
   git tag v0.0.1
   git push origin v0.0.1
   ```

The workflow builds with:

```powershell
python -m build
```

It publishes to PyPI with:

```text
pypa/gh-action-pypi-publish@release/v1
```

Users install the package with:

```powershell
python -m pip install py3dengine
```

and import it with:

```python
import py_3d
```

## Optional py_gpu package

The optional GPU bridge package lives in the separate sibling repository
`py_gpu` and publishes as `py_gpu`.

Trusted Publisher values:

| PyPI field | Value |
| --- | --- |
| PyPI Project Name | `py_gpu` |
| Owner | `Daniel-J-Mueller` |
| Repository name | `py_gpu` |
| Workflow name | `publish.yml` |
| Environment name | any / blank |

Release flow:

1. Update `version` in the `py_gpu` repository's `pyproject.toml`.
2. Commit and push the release changes.
3. Create and push a release tag:

   ```powershell
   git tag v0.0.1
   git push origin v0.0.1
   ```

The workflow builds with:

```powershell
python -m build
```

Users install the optional bridge with:

```powershell
python -m pip install py_gpu
```
