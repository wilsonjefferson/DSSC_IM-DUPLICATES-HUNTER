import unittest
from parameterized import parameterized

from core.FIleHandler import FileHandler
from exceptions.FileNotExistException import FileNotExistException
from exceptions.HeaderNotExistWarning import HeaderNotExistWarning


class FileHandlerTest(unittest.TestCase):

    def tearDown(self):
        pass

    def setUp(self):
        self.file_handler = FileHandler()

    @parameterized.expand([
        ('FileHandlerTest_Test_01', '\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\Incidents 2019-2022.xlsx'),
    ])
    def test_check_file1(self, _, test_absolute_path):
        self.assertTrue(self.file_handler.check_file(test_absolute_path))

    @parameterized.expand([
        ('FileHandlerTest_Test_02', '\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\fake.xlsx', FileNotExistException),
    ])
    def test_check_file2(self, _, test_absolute_path, test_exception):
        self.assertRaises(test_exception, self.file_handler.check_file, test_absolute_path)
    
    @parameterized.expand([
        ('FileHandlerTest_Test_03', '\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\Incidents 2019-2022.xlsx', ['Incident ID', 'Title']),
    ])
    def test_get_columns1(self, _, source_file, expected):
        self.file_handler.source_file = source_file
        self.assertEqual(expected, self.file_handler.get_columns())

    @parameterized.expand([
        ('FileHandlerTest_Test_04', '\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\Incidents 2019-2022.xlsx', HeaderNotExistWarning),
    ])
    def test_get_columns2(self, _, source_file, test_exception):
        self.file_handler.source_file = source_file
        self.assertRaises(test_exception, self.file_handler.get_columns)


if __name__ == '__main__':
    unittest.main()