import unittest
import wisdem.commonse.enum as enum

class TestEnum(unittest.TestCase):
    def setUp(self):
        self.myenum = enum.Enum("One two THREE")

    def testValues(self):
        self.assertEqual(self.myenum.ONE, 0)
        self.assertEqual(self.myenum.TWO, 1)
        self.assertEqual(self.myenum.THREE, 2)

    def testValueFromString(self):
        self.assertEqual(self.myenum['ONE'], 0)
        self.assertEqual(self.myenum['TWO'], 1)
        self.assertEqual(self.myenum['THREE'], 2)

    def testValueFromUnicode(self):
        self.assertEqual(self.myenum[u'ONE'], 0)
        self.assertEqual(self.myenum[u'TWO'], 1)
        self.assertEqual(self.myenum[u'THREE'], 2)

    def testLen(self):
        self.assertEqual( len(self.myenum), 3)

    def testReverseLookup(self):
        self.assertEqual( self.myenum[0], 'ONE')
        self.assertEqual( self.myenum[1], 'TWO')
        self.assertEqual( self.myenum[2], 'THREE')
        self.assertEqual( self.myenum[-1], 'THREE')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestEnum))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
