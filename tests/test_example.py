from pyCarDisplay.pyCarDisplay import CarDisplay
import unittest


class BColors:
    """Colors for printing"""
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    
    
class TestExample(unittest.TestCase):
    
    def setUp(self):
        """Setup the test."""
        self.test = True
        
        
    def tearDown(self):
        """End of testing routines."""
        pass
    
    
    def test_example(self):
        """Ensure the sample size matches the expected."""

        self.assertEqual(self.test, True, 
                         msg=f'{BColors.FAIL}\t[-]\tTest failed!{BColors.ENDC}')
        print(f"{BColors.OKGREEN}\t[+]\tTest passes!{BColors.ENDC}")