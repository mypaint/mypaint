import unittest
from glob import glob
from lib import localecodes as lc


class LocaleTest(unittest.TestCase):

    def test_langcodes(self):
        """Verify existence of ui strings for all codes
        """
        modname = "lib.localecodes"
        self.assertEqual(
            lc.__name__, modname,
            msg="locale code module has been renamed, update test messages!"
        )
        code_list_varname = "LOCALE_DICT"
        self.assertTrue(
            hasattr(lc, code_list_varname),
            msg="locale code dict var has been renamed, update test messages!"
        )
        fullname = modname + "." + code_list_varname
        locdir = "po/"
        codes = [c[len(locdir):-len('.po')] for c in glob(locdir + "*.po")]
        self.assertTrue(
            all([c in lc.LOCALE_DICT for c in codes]),
            msg="The following codes do not have entries in {dictname}:\n"
            "{codes}".format(
                dictname=fullname,
                codes="\n".join([c for c in codes if c not in lc.LOCALE_DICT])
            )
        )
