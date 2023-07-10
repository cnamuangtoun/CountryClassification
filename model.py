from torch import nn



class CNN_model(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 27, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(27, 81, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(81, 243, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc1 = nn.Linear(243*40*40, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 21)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.fc1(out.view(out.size(0), 243*40*40).squeeze())
        out = self.relu(out)
        out = self.fc2(out)
        out = nn.LogSoftmax(dim=1)(out)
        return out
