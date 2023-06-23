import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

GROUP_SIZE = 2


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