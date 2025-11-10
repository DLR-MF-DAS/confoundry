# Generic Python Template

A generic Python project template for MF-DAS. Use this for "standard" Python packages.

After initializing your project from this template I would advise to do the following:

* Enable GitHub Pages by going to `Settings` -> `Pages` and under `Build and deployment` subsection `Source` choose `GitHub Actions` instead of `Deploy from a branch`.
* Change the name of your package by checking out the project and doing `git mv src/project_name src/your_actual_project_name` where `your_actual_project_name` is the name you chose for your project. After that `git commit -am "change package name"` and `git push`.
* Change the project name and other information in the `pyproject.toml`. Commit changes with `git commit -am "updated pyproject.toml"` and `git push`.
* Open .githubt/workflows/pdoc.yml in an editor and change `project_name` to the proper project name. Commit changes with `git commit -am "updated documentation build pipeline"` and `git push`.

# GitHub Actions

CI/CD tasks for your project will be handled by GitHub Actions which are CI/CD pipelines described in a YAML-based language. Here we have a list of them that you'll need to include in your project and ensure they are working correctly.

## Testing

Taken care of in `.github/workflows/python-app.yml`. By default `pytest` is used. Find tests under `test/` and test your code religiously. No untested or undocumented code is allowed.

## Documentation Building

Taken care of in `.github/workflows/pdoc.yml`. The workflow will take care of building and publishing project documentation from Python docstrings. You need to use the `numpy` docstring format. Find the description of it [here](https://numpydoc.readthedocs.io/en/latest/format.html). The documentation will be built and published under dlr-mf-das.github.io/general-template where you need to replace `general-template` with your repository name.

## Python Package Publishing

Taken care of in `.github/workflows/python-publish.yml`. This workflow is executed every time a new release is created. Make sure to update the version number in `pyproject.toml` before creating a release. This number is used to create the Python package archive published to MF-DAS pypi. If the package with that version number was previously uploaded to PyPi this workflow will fail.
