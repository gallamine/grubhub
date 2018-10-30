Grubhub Model Trainer
=====================

[![image]]

[![image][1]]

[![Documentation Status]]

Grubhub MLE takehome model trainer.

This provides a command-line interface to train a model.

-   Free software: MIT license
-   Documentation: <https://grubhub.readthedocs.io>.

Local Development
-----------------

Create a `conda` environment for development with `conda env create`. Then `source activate grubhub` to activate it.

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
-   Define multiple data sources under `data/training_data.json`
-   Build models from cached data (not implemented yet)

Credits
-------

Written by William Cox

This package was created with [Cookiecutter] and the
[audreyr/cookiecutter-pypackage] project template.

  [image]: https://img.shields.io/pypi/v/grubhub.svg
  [![image]]: https://pypi.python.org/pypi/grubhub
  [1]: https://img.shields.io/travis/gallamine/grubhub.svg
  [![image][1]]: https://travis-ci.org/gallamine/grubhub
  [Documentation Status]: https://readthedocs.org/projects/grubhub/badge/?version=latest
  [![Documentation Status]]: https://grubhub.readthedocs.io/en/latest/?badge=latest
  [Cookiecutter]: https://github.com/audreyr/cookiecutter
  [audreyr/cookiecutter-pypackage]: https://github.com/audreyr/cookiecutter-pypackage
