.. how_to_contribute_code:

How to contribute code to WISDEM
================================

.. note::
  This section will be expanded in the future.

WISDEM is an open-source tool, thus we welcome users to submit additions or fixes to the code to make it better for everybody.

Issues
------
If you have an issue with WISDEM, a bug to report, or a feature to request, please submit an issue on the GitHub repository.
This lets other users know about the issue.
If you are comfortable fixing the issue, please do so and submit a pull request.

Documentation
-------------
When you add or modify code, make sure to provide relevant documentation that explains the new code.
This should be done in code via comments, but also in the Sphinx documentation as well if you add a new feature or capability.
Look at the .rst files in the `docs` section of the repo or click on `view source` on any of the doc pages to see some examples.

Testing
-------
When you add code or functionality, add tests that cover the new or modified code.
These may be units tests for individual components or regression tests for entire models that use the new functionality.

Each discipline sub-directory contains tests in the `test` folder.
For example, `wisdem/test/test_ccblade` hosts the tests for CCBlade within WISDEM.
Look at `test_ccblade.py` within that folder for a simple unit test that you can mimic when you add new components.

Pull requests
-------------
Once you have added or modified code, submit a pull request via the GitHub interface.
This will automatically go through all of the tests in the repo to make sure everything is functioning properly.
This also automatically does a coverage test to ensure that any added code is covered in a test.
The main developers of WISDEM will then merge in the request or provide feedback on how to improve the contribution.
