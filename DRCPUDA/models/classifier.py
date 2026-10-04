import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dims, output_dims, bias=False),
        )

    def forward(self, x):
        out = self.fc(x)
        return out


def classifier(input_dims, output_dims):
    return Classifier(input_dims, output_dims)
