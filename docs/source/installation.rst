Installation
============

Requirements
------------

Python 3.10–3.13.

User install
------------

.. code-block:: bash

   pip install djura

Optional runtime extras
-----------------------

.. code-block:: bash

   pip install "djura[plot]"    # matplotlib
   pip install "djura[hdf5]"    # h5py
   pip install "djura[all]"     # all of the above

Contributor install
-------------------

Clone the repository and install with Poetry:

.. code-block:: bash

   git clone https://github.com/djura-risk-data-engineering/djura.git
   cd djura
   poetry install --with dev         # testing + linting
   poetry install --with docs        # Sphinx documentation tools
   poetry install --with dev,docs    # everything

.. note::

   ``sphinx-autodoc-typehints`` in the ``docs`` group requires Python ≥ 3.12
   and is skipped automatically on earlier versions.
