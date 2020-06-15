import unittest
import pytest

import wisdem.test.test_all as test_all
import os
import sys

testpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + 'wisdem' + os.path.sep + 'test' + os.path.sep

mytests = [testpath + m for m in test_all.valid_tests]

if __name__ == '__main__':
    ret = pytest.main(mytests)
    sys.exit(ret)
    #unittest.TextTestRunner().run(test_all.suite())
