Grubhub Model Trainer
=====================

[![image](https://img.shields.io/pypi/v/grubhub.svg)](https://pypi.python.org/pypi/grubhub)

[![travis](https://img.shields.io/travis/gallamine/grubhub.svg)](https://travis-ci.org/gallamine/grubhub)

[![Documentation Status](https://readthedocs.org/projects/grubhub/badge/?version=latest)](https://readthedocs.org/projects/grubhub/badge/?version=latest)

Grubhub MLE takehome model trainer.

This provides a command-line interface to train a model.

-   Free software: MIT license
-   Documentation: <https://grubhub.readthedocs.io>.

Local Development
-----------------

Check out this repository:

```bash
git clone git@github.com:gallamine/grubhub.git
```

Create a [`conda`](https://conda.io/miniconda.html) environment for development with `conda env create`. Then `source activate grubhub` to activate it.

Install the package for local development:
```
make install-dev
```

Use
---

After installing the package via `make install` you can call the CLI via `grubhub`.

Features
--------

-   Train a model for specific regions using the `-r` or `--region` cli argument
-   Define multiple data sources under `grubhub/training_data.py`
-   Build models from cached data (not implemented yet)

Credits
-------

Written by William Cox

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
