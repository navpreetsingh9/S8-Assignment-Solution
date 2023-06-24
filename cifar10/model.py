import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

GROUP_SIZE = 2
dropout_value = 0.05


class Block(nn.Module):
    def __init__(self, input_size, output_size, filter1x, padding=1, norm='bn', usepool=True, usefilter1x=True):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm (str, optional): Type of normalization to be used. Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.
            usepool (bool, optional): Enable/Disable Maxpolling. Defaults to True.
        """
        super(Block, self).__init__()
        self.usepool = usepool
        self.usefilter1x = usefilter1x
        self.conv1 = nn.Conv2d(input_size, output_size, 3, padding=padding)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(output_size)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, output_size)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, output_size)
        self.conv2 = nn.Conv2d(output_size, output_size, 3, padding=padding)
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(output_size)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, output_size)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, output_size)
        self.conv3 = nn.Conv2d(output_size, output_size, 3, padding=padding)
        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(output_size)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(GROUP_SIZE, output_size)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, output_size)
        if usefilter1x:
            self.filter1x = nn.Conv2d(output_size, filter1x, 1)
        if usepool:
            self.pool = nn.MaxPool2d(2, 2)

    def __call__(self, x, layers=3, first=False, last=False):
        """
        Args:
            x (tensor): Input tensor to this block
            layers (int, optional): Number of layers in this block. Defaults to 3.
            last (bool, optional): Is this the last block. Defaults to False.

        Returns:
            tensor: Return processed tensor
        """
        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)
        if not first:
            out = x.clone()
        x = self.conv2(x)
        x = self.n2(x)
        x = F.relu(x)
        if layers >= 3:
            x = self.conv3(x)
            x = self.n3(x)
            x = F.relu(x)
        if not first:
            x = x + out
        if self.usefilter1x:
            x = self.filter1x(x)
        if self.usepool:
            x = self.pool(x)
        return x


class Net(nn.Module):
    """ Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(self, base_channels=16, filter_channels=8, layers=3, drop=0.01, norm='bn'):
        """Initialize Network

        Args:
            base_channels (int, optional): Number of base channels to start with. Defaults to 4.
            layers (int, optional): Number of Layers in each block. Defaults to 3.
            drop (float, optional): Dropout value. Defaults to 0.01.
            norm (str, optional): Normalization type. Defaults to 'bn'.
        """
        super(Net, self).__init__()

        self.base_channels = base_channels
        self.filter_channels = filter_channels
        self.drop = drop
        self.no_layers = layers

        # Conv
        self.block1 = Block(3, self.base_channels, self.filter_channels, norm=norm)
        self.dropout1 = nn.Dropout(self.drop)
        self.block2 = Block(self.filter_channels,
                            self.base_channels*2, self.filter_channels*2, norm=norm)
        self.dropout2 = nn.Dropout(self.drop)
        self.block3 = Block(self.filter_channels*2,
                            self.base_channels*2, self.filter_channels*2, norm=norm,
                            usepool=False, usefilter1x=False)
        self.dropout3 = nn.Dropout(self.drop)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Conv2d(self.base_channels*2, 10, 1)

    def forward(self, x, dropout=True):
        """Convolution function

        Args:
            x (tensor): Input image tensor
            dropout (bool, optional): Enable/Disable Dropout. Defaults to True.

        Returns:
            tensor: tensor of logits
        """
        # Conv Layer
        x = self.block1(x, layers=2, first=True)
        if dropout:
            x = self.dropout1(x)
        x = self.block2(x, layers=3)
        if dropout:
            x = self.dropout2(x)
        x = self.block3(x, layers=3, last=True)

        # Output Layer
        x = self.gap(x)
        x = self.flat(x)
        x = x.view(-1, 10)

        # Output Layer
        return F.log_softmax(x, dim=1)



class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(64, 128, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(128, 256, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 | 
        self.conv7 = nn.Conv2d(256, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), # 28>28 | 1>3 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 8, 3, padding=1), # 28>28 | 3>5 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 8, 3, padding=1), # 28>28 | 5>7 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2)  # 28>14 | 7>8 | 1>2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 13, 3, padding=1), # 14>14 | 8>12 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.Conv2d(13, 13, 3, padding=1), # 14>14 | 12>16 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.Conv2d(13, 13, 3, padding=1), # 14>14 | 16>20 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2) # 14>7 | 20>22 | 2>4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(13, 18, 3), # 7>5 | 22>30 | 4>4
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(dropout_value),
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(3), # 5>1 | 30>38 | 4>12
            nn.Conv2d(18, 10, 1) # 1>1 | 38>38 | 12>12
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=1)
        return x        


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 7, 3, padding=1), # 28>28 | 1>3 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(7),
            nn.Dropout(dropout_value),
            nn.Conv2d(7, 7, 3, padding=1), # 28>28 | 3>5 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(7),
            nn.Dropout(dropout_value),
            nn.Conv2d(7, 7, 3, padding=1), # 28>28 | 5>7 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(7),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2)  # 28>14 | 7>8 | 1>2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(7, 10, 3, padding=1), # 14>14 | 8>12 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.Conv2d(10, 10, 3, padding=1), # 14>14 | 12>16 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.Conv2d(10, 10, 3, padding=1), # 14>14 | 16>20 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2) # 14>7 | 20>22 | 2>4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 13, 3, padding=1), # 7>7 | 22>30 | 4>4
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.Conv2d(13, 13, 3, padding=1), # 7>7 | 30>38 | 4>4
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2), # 7>3 | 38>42 | 4>8
            nn.Conv2d(13, 13, 3, padding=1), # 3>3 | 42>58 | 8>8
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(3), # 3>1 | 58>74 | 8>24
            nn.Conv2d(13, 10, 1) # 1>1 | 74>74 | 24>24
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=1)
        return x            