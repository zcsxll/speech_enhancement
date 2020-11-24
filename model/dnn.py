import torch

class Model(torch.nn.Module):
    def __init__(self, **args):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=64000, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=512)
        self.fc3 = torch.nn.Linear(in_features=512, out_features=512)
        self.fc4 = torch.nn.Linear(in_features=512, out_features=512)
        self.fc5 = torch.nn.Linear(in_features=512, out_features=512)
        self.fc6 = torch.nn.Linear(in_features=512, out_features=64000)

    def forward(self, x):
        if x.dtype == torch.int16:
            x = x / 32767.0
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)
        x = torch.nn.ReLU()(x)
        x = self.fc3(x)
        x = torch.nn.ReLU()(x)
        x = self.fc4(x)
        x = torch.nn.ReLU()(x)
        x = self.fc5(x)
        x = torch.nn.ReLU()(x)
        x = self.fc6(x)
        x = torch.nn.Sigmoid()(x)
        return x, None

    def total_parameter(self):
        return sum([p.numel() for p in self.parameters()])

if __name__ == '__main__':
    model = Model()
    x = torch.randn((4, 256))
    print(x.shape)
    ret = model(x)
    print(ret.shape)
