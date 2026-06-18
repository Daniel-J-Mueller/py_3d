# Publishing

This repository publishes as `py3dengine` and imports as `py_3d`.

The PyPI distribution name in `pyproject.toml` is `py3dengine`. Users install
that distribution name with `pip`, then import the Python package as `py_3d`.
The names do not need to match.

Use this Trusted Publisher configuration on PyPI:

| PyPI field | Value |
| --- | --- |
| PyPI Project Name | `py3dengine` |
| Owner | `Daniel-J-Mueller` |
| Repository name | `py_3d` |
| Workflow name | `publish.yml` |
| Environment name | any |

First release flow:

1. Create or claim the PyPI project `py3dengine`.
2. Add a PyPI Trusted Publisher for GitHub Actions with the values above.
3. Commit `.github/workflows/publish.yml`.
4. Push the repository to GitHub.
5. Create and push a release tag:

   ```powershell
   git tag v0.0.1
   git push origin v0.0.1
   ```

The workflow builds with:

```powershell
python -m build
```

It publishes with:

```text
pypa/gh-action-pypi-publish@release/v1
```

After publishing, users install the package with:

```powershell
python -m pip install py3dengine
```

and import it with:

```python
import py_3d
```
