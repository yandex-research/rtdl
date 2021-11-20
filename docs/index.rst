Revisiting Tabular Deep Learning
================================

:code:`rtdl` (Revisiting Tabular Deep Learning) is a Python package based on the `implementation <https://github.com/yandex-research/rtdl>`_
of the paper "Revisiting Deep Learning Models for Tabular Data":

- It can be used by practitioners looking for Deep Learning models for tabular data
- It can serve as a source of baselines for researchers (**excluding FT-Transformer**, see the warning below)
- You can follow releases by hitting "Watch" / "Custom" / "Releases" in the right upper corner of the GitHub interface

.. warning:: If you are a *researcher* (not a practitioner) and plan to use the
   FT-Transformer model as a baseline in your paper, please, use the original implementation
   from `ft_transformer.py <https://github.com/yandex-research/rtdl/blob/main/bin/ft_transformer.py>`_.
   We will remove this limitation soon (i.e. :code:`rtdl` will become the recommended way to
   use FT-Transformer in papers).

Users can share their experience here: https://github.com/yandex-research/rtdl/discussions/1

Installation
------------

.. code-block::

   pip install rtdl

Documentation
-------------

- See the API reference in the left sidebar
- Colab example: https://colab.research.google.com/github/yandex-research/rtdl/blob/main/examples/rtdl.ipynb

.. toctree::
   :caption: API REFERENCE
   :hidden:

   rtdl
   rtdl_data

----

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
