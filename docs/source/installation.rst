Installation
============

Requirements
------------

Python 3.10–3.13.

User install
------------

djura is organised as a set of applications that can be installed
independently. A bare install provides only the shared core (``numpy``,
``scipy``, ``pydantic``):

.. code-block:: bash

   pip install djura

Per-application installs
------------------------

Each application declares its own dependencies under an extra named after
the submodule:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Install command
     - Provides
   * - ``pip install "djura[record_selection]"``
     - :mod:`djura.record_selection`
   * - ``pip install "djura[hazard_consistency]"``
     - :mod:`djura.hazard_consistency`
   * - ``pip install "djura[edp_im]"``
     - :mod:`djura.edp_im`
   * - ``pip install "djura[im_conversion]"``
     - :mod:`djura.im_conversion`
   * - ``pip install "djura[vulnerability_modeller]"``
     - :mod:`djura.vulnerability_modeller`
   * - ``pip install "djura[slf]"``
     - :mod:`djura.slf`

.. note::

   ``im_conversion`` was named ``fragility_converter`` before 2.0.1, since it
   converts vulnerability models as well as fragility ones. The old extra and
   the ``djura.fragility_converter`` import path still work — the latter with
   a :class:`DeprecationWarning` — and are removed in 3.0.

Extras are additive, so several applications can be requested together:

.. code-block:: bash

   pip install "djura[record_selection,slf]"

Everything at once
------------------

.. code-block:: bash

   pip install "djura[all]"

Optional accelerators
---------------------

.. code-block:: bash

   pip install "djura[record_selection,hdf5]"   # h5py, for GMPE tables
   pip install "djura[edp_im,xgboost]"          # gradient-boosted models

.. note::

   **Upgrading from 1.x.** A bare ``pip install djura`` no longer installs
   every application's dependencies. Use ``pip install "djura[all]"`` to
   keep the previous behaviour, or name only the applications you use.
   Importing an application whose extra is not installed raises an
   ``ImportError`` naming the exact command to run.

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
