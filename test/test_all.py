import os
import pytest

testpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + \
    'wisdem' + os.path.sep + 'test' + os.path.sep

sys.exit( pytest.main([testpath] )

