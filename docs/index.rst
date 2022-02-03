.. Leniax documentation main file

Leniax documentation
====================

Leniax is a `Lenia <https://chakazul.github.io/lenia.html>`_ simulation library powered by JAX.
It can efficiently simulate and render Lenia worlds and can also be used to search for creatures, initial conditions, kernels and growth functions.
It is also fully differentiable!

For a quick introduction and short example snippets, see our `README <https://github.com/morgangiraud/leniax/blob/main/README.md>`_.

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   overview
   installation
   Examples <https://github.com/morgangiraud/leniax/tree/main/examples>
   Experiments <https://github.com/morgangiraud/leniax/tree/main/experiments>

.. toctree::
   :maxdepth: 1
   :caption: Guided Tour

   Running a simple simulation <https://colab.research.google.com/drive/1-UAq3_QzhFMrn341-eMCeTVnDKshTqrv>
   Running multiple initializations simulation <https://colab.research.google.com/drive/1Mn4gT_tcq23lGRYHI_g4cT74bUXiEsKi>
   Running multiple simulations <https://colab.research.google.com/drive/1h38u8Al44aZzN9FHIR8rBfq8Wlnm8vtD>
   Quality-Diversity search <https://colab.research.google.com/drive/1-UAq3_QzhFMrn341-eMCeTVnDKshTqrv>
   Learning a simulation parameters <https://colab.research.google.com/drive/1a6bVjpX8AijE5FBqV5XEegZclq10DMdt>


.. toctree::
   :maxdepth: 1
   :caption: Additional material

   philosophy
   contributing


.. toctree::
   :maxdepth: 2
   :caption: API reference

   leniax.core
   leniax.runner
   leniax.growth_functions
   leniax.kernel_functions
   leniax.helpers
   leniax.utils
   leniax.video
   leniax.colormaps

