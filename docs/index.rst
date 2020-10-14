.. qsearch documentation master file, created by
   sphinx-quickstart on Wed Oct 14 05:28:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qsearch's documentation!
===================================

.. note::
   The documentation is currently a work in progress and will be expanded upon soon.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gatesets
   gates

Working with nonlinear topologies
+++++++++++++++++++++++++++++++++
The default topology is linear.  To synthesize for another topology, you will need to choose
a gateset for your desired topology, usually either `QubitCNOTRing` or `QubitCNOTAdjacencyList`,
but custom gatesets are also supported.  See :doc:`gatesets` for more information.

Working with nonstandard gates or qutrits
+++++++++++++++++++++++++++++++++++++++++
You will to choose a gateset that supports your desired gates.  See :doc:`gatesets` for a list
of implemented gatesets, and instructions on how to make your own.  See :doc:`gates` for a
list of supported gates and instructions on how to make your own.

Customizing your compilation
++++++++++++++++++++++++++++
Once you have your desired gateset object, you can pass it either to a `Project` or a
`SearchCompiler`.  In addition, both `Project` and `SearchCompiler` have many other
options used for customizing things like the search type or distance function.
See the `Options` API documentation for more information.

Make sure to check out the `example scripts <https://github.com/BQSKit/qsearch/tree/master/examples>`_ as well!

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
