import unittest
from unittest import TestCase

from src.Application import Application
from src.exception.InvalidActionError import InvalidActionError


class TestApplication(TestCase):
    def setUp(self):
        self.application = Application()

    def test_step_invalid_action(self):
        with self.assertRaises(InvalidActionError):
            self.application.step('action')

    def test_reset(self):
        self.fail()

    def test_close(self):
        self.fail()


if __name__ == '__main__':
    unittest.main()
