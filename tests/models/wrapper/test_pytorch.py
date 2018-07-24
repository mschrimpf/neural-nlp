import numpy as np
from torch import nn

from neural_nlp.models.wrapper.pytorch import PytorchModel


class TestPytorchModel:
    def test_available_layers(self):
        class InnerModule(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 10)
                self.fc2 = nn.Linear(10, output_size)

            def forward(self, input):
                fc1 = self.fc1(input)
                relu1 = nn.ReLU(fc1)
                fc2 = self.fc2(relu1)
                relu2 = nn.relu(fc2)
                return relu2

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = InnerModule(100, 10)
                self.m2 = InnerModule(10, 2)

            def forward(self, input):
                m1 = self.m1(input)
                m2 = self.m2(m1)
                return m2

        class ModelWrapper(PytorchModel):
            def _load_model(self):
                return Model()

        model = ModelWrapper()
        layers = model.available_layers()
        np.testing.assert_array_equal(['m1.fc1', 'm1.fc2', 'm2.fc1', 'm2.fc2'], layers)
