import os
import sys
import pytest

testpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + \
    'wisdem' + os.path.sep + 'test' + os.path.sep

if __name__ == '__main__':
    sys.exit( pytest.main([testpath]) )
