import sys
from pathlib import Path
from warnings import warn

import pytest

testpath = Path(__file__).parents[1] / "wisdem/test"

if __name__ == "__main__":
    warn("In a future version this routine will be removed. Please run 'pytest' to run all tests.", DeprecationWarning)
    sys.exit(pytest.main(testpath))
