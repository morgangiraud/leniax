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

   Running a simple simulation <https://colab.research.google.com/drive/1Z0dBQrENxqcDdCGvpFzx0xwLobU_uDA1>
   Running multiple initializations simulation <https://colab.research.google.com/drive/1Mn4gT_tcq23lGRYHI_g4cT74bUXiEsKi>
   Running multiple simulations <https://colab.research.google.com/drive/1h38u8Al44aZzN9FHIR8rBfq8Wlnm8vtD>
   Initialization search (TPU) <https://colab.research.google.com/drive/1Kk1GXUGD6GiCdMy_fgL6LrwR2doDbXP_>
   Initialization search (GPU) <https://colab.research.google.com/drive/1vIOfp3zJa5GLYevE_W3eOS1BJ-FOfegd>
   Quality-Diversity search <https://colab.research.google.com/drive/1-UAq3_QzhFMrn341-eMCeTVnDKshTqrv>
   Learning a simulation parameters <https://colab.research.google.com/drive/1a6bVjpX8AijE5FBqV5XEegZclq10DMdt>
   Neural Lenia <https://colab.research.google.com/drive/1GU7aeGPUU1bH932Y27Aw6-LV3YoOBkkE>
   Neural CA <https://colab.research.google.com/drive/1c8A7kYiY8Ew2qqp6ksCgna5a5thEqQyv>
   Conway's game of life <https://colab.research.google.com/drive/1X49aI4GXnlK76i4X5rB8Mz_xshsjs6nu>


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
   leniax.kernels
   leniax.kernel_functions
   leniax.statistics   
   leniax.helpers
   leniax.utils
   leniax.loader
   leniax.qd
   leniax.lenia
   leniax.video
   leniax.colormaps

