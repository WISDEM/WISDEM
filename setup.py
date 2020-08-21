from setuptools import setup

from os import path
from io import open

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

CLASSIFIERS = """
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
"""

metadata = dict(
    name="weis",
    version="0.0.1",
    description="Wind Energy with Integrated Servo-control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NREL",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    packages=["weis"],
    python_requires=">=3.6",
    zip_safe=True,
)

setup(**metadata)