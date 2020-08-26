.. how_to_write_docs:

How to write docs for WEIS code
===============================

.. TODO: expand this

Introduction
------------
This page describes how to add, improve, and update any of the WEIS documentation.
The WEIS documentation can be divided into two categories, a documentation for WEIS submodules, and then WEIS usage documentation, tutorials and guides.


Getting started with the docs
-----------------------------

When adding or updating the documentation please try to follow the these guidelines:

* files and folders should follow the existing naming convention
* images, figures and files should be placed in specific folders

To update the repo, you need to commit and push your changes. Use the following commands, but with a more descriptive commit message::

    git commit -am "Updated docs"
    git push

Sphinx and rst
--------------
In all cases documentation is generated using the `Sphinx <http://www.sphinx-doc.org/en/master/index.html>`_ documentation generator.

The source files or the documentation itself is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`_. A primer on the ``rst`` syntax can be found `here <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_. In general http://www.sphinx-doc.org is very helpful for syntax and examples.

.. NOTE::
    When viewing the documentation in a browser you can always view the source by clicking the **Show Source** link. This is also a great way of getting examples.

The sphinx system uses Makefiles to generate the documentation from the source ``.rst`` files.

In any case, to build the documentation, navigate to the ``docs`` folder and run the following command from the command line::

    make html



General guidelines for formatting
---------------------------------

Headings
~~~~~~~~
When contributing to any documentation please use the following character for heading levels::

    Sample heading 1
    ================

    Sample heading 2
    ----------------

    Sample heading 3
    ~~~~~~~~~~~~~~~~

    Sample heading 4
    ****************


.. NOTE::
    Make sure the character underlines the entire heading or Sphinx will generate warnings when building.

Tables
~~~~~~
Tables can be difficult to get "right" in html and especially when compiled to LaTeX. Using the simple version of tables often leads to imbalanced column widths and building LaTeX documents often results in bad tables. To try to mitigate this issue another table type should be used::

    .. tabularcolumns:: |>{\raggedright\arraybackslash}\X{1}{5}|>{\raggedright\arraybackslash}\X{1}{5}|>{\raggedright\arraybackslash}\X{3}{5}|

    .. list-table:: Demo table title
        :widths: 15 20 65
        :header-rows: 1

        * - Col 1
          - Col 2
          - Col 3

        * - Entry 1
          - Entry 2
          - Entry 3

.. NOTE::

    - ``tabularcolumns``: Controls how LaTeX generates the following table. The ``widths`` keyword is overridden/omitted for LaTeX when this keyword is specified. The \X{1}{5} is this column width ratio.
    - ``widths``: keyword represents columns widths in percentages.
    - ``header-rows`` keyword specifies how many rows are made bold and shaded


The code above generates the following table

.. list-table:: Demo table title
    :widths: 15 20 65
    :header-rows: 1

    * - Col 1
      - Col 2
      - Col 3

    * - Entry 1
      - Entry 2
      - Entry 3



Where should files related to documentation live?
-------------------------------------------------
Add figures and small files to the ``docs`` folder then embed them as desired
in the `.rst` files.
For larger files, host them elsewhere and link to them from the docs.


Where should you contribute docs to?
------------------------------------
As you begin to determine if you should write a doc page, first search for relevant
entries to make sure you don't duplicate something that already exists.
You can then judge if your contribution should be in its own doc page or if it
should be added to an existing page.
Make sure to think logically about where the information you prepared should live so
it is intuitive for other people, especially people just starting out.

Once you have added your `.rst` file in the repo in a logical place within the file
structure, also update the table of contents in the other relevant `.rst` files as
needed.
This ensures that your contributions can be easily found.

How to convert existing docs
----------------------------
If you already have something typed up, either in Latex, a basic text file, or another
format, it's usually pretty straightforward to convert this to rst.
`Pandoc <https://pandoc.org/demos.html>`__ is a helpful automated tool that converts
text files near seamlessly.

How to request doc creation
---------------------------
If you think the docs should be modified or expanded, create an issue on the GitHub documentation repository.
Do this by going to the `WEIS repo <https://github.com/WISDEM/WEIS/>`__ then click on Issues on the lefthand side of the page.
There you can see current requests for doc additions as well as adding your own.
Feel free to add any issue for any type of doc and members of the WEIS development team can determine how to approach it.
Assign someone or a few people to the issue who you think would be a good fit for that doc.
