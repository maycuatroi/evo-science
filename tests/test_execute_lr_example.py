import unittest
import sys

from examples.titanic_survival_prediction.linear_regression_model import (
    example_lr_model,
)

# appending the needed path for import
sys.path.append("/evo_science")
sys.path.append("/examples/titanic-survival-prediction")


class TestExample(unittest.TestCase):
    def test_execute_lr_example(self):
        # set working directory
        example_lr_model()
