Contributing
============

Contributions are welcome. By submitting a pull request you agree to the
terms of the `Contributor License Agreement <https://github.com/djura-risk-data-engineering/djura/blob/main/CLA.md>`_.

Setting up a development environment
-------------------------------------

.. code-block:: bash

   git clone https://github.com/djura-risk-data-engineering/djura.git
   cd djura
   poetry install --with dev

Running the test suite
----------------------

.. code-block:: bash

   pytest

   # skip slow tests (they require a local flatfile)
   pytest -m "not slow"

   # run slow tests locally (bash) — the flatfile is provided by the user
   export DJURA_METADATA_PATH=/path/to/flatfile.pickle
   pytest -m slow

   # run slow tests locally (PowerShell)
   $env:DJURA_METADATA_PATH = "C:\path\to\flatfile.pickle"
   pytest -m slow

Building the documentation
--------------------------

.. code-block:: bash

   poetry install --with docs
   cd docs
   make html
   # open docs/build/html/index.html

Code style
----------

The project uses ``flake8`` for linting (max line length 79):

.. code-block:: bash

   flake8 src/djura
