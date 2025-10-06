Installation
============

This guide covers how to install ``bard`` for both regular use and for development.

User Installation
-----------------

To install the core ``bard`` library, first clone the repository from GitHub and then install it in editable mode using pip.

.. code-block:: bash

   git clone https://github.com/YueWang996/bard.git
   cd bard
   pip install -e .

This installs the library and its essential dependencies (``torch`` and ``numpy``). The ``-e`` flag ensures that any changes you make to the source code are immediately available in your environment.

Developer Installation
----------------------

If you plan to contribute to ``bard`` or run the test suite, you need to install the development dependencies.

.. code-block:: bash

   # Run this from the root of the project
   pip install -e .[dev]

This command installs everything from the user installation, plus additional tools required for testing, such as ``pytest`` and ``pinocchio``.


Building the Documentation
--------------------------

To build the HTML documentation locally, you first need to install the documentation-specific dependencies.

1. **Install Documentation Tools**

.. code-block:: bash

   # Run this from the root of the project
   pip install -e .[docs]


2. **Generate the HTML Files**

   Navigate into the ``docs`` directory and use the provided Makefile to build the site.

.. code-block:: bash

   cd docs
   make html

3. **View the Documentation**

   The generated site will be in the ``docs/build/html/`` directory. You can open the ``index.html`` file in that folder with your web browser.