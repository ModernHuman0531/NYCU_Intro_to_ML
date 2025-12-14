import torch
import torch.nn as nn
import torch.nn.functional as F

# Define class output number as 10 for FashionMNIST
NUM_CLASSES=10
# Define graph input size as 28*28
INPUT_SIZE=28*28

# --- 1. Baseline Fully Connected Network (NN/MLP) ---
class BaselineNN(nn.Module):
    """
    Simple MLP, with two hidden layers, activation function ReLU.
    """
    def __init__(self, hidden_dim1=512, hidden_dim2=256):
        super(BaselineNN, self).__init__()

        # 1. First hidden layer: INPUT_SIZE -> hidden_dim1
        self.fc1=nn.Linear(INPUT_SIZE,hidden_dim1)
        # 2. Second hidden layer: hidden_dim1 -> hidden_dim2
        self.fc2=nn.Linear(hidden_dim1,hidden_dim2)
        # 3. Output layer: hidden_dim2 -> NUM_CLASSES
        self.fc3=nn.Linear(hidden_dim2,NUM_CLASSES)

        # Record structure information to calculate amount of parameters
        self.param_dims=[(INPUT_SIZE,hidden_dim1),
                         (hidden_dim1,hidden_dim2),
                         (hidden_dim2,NUM_CLASSES)]
        
    def forward(self, x):
        """
        Forward process of BaselineNN.
        Architecture: INPUT_SIZE -> ReLU -> Linear -> ReLU -> Linear -> Softmax (in loss function)
        No BN/Dropout layers.
        """

        # H1: Linear -> ReLU
        x=F.relu(self.fc1(x))
        # H2: Linear -> ReLU
        x=F.relu(self.fc2(x))
        # Output layer: Linear
        x=self.fc3(x)

        # Noted that Softmax is included in CrossEntropyLoss, so no need to add here
        return x
    
# --- 2. Improved Fully Connected Network (NN/MLP), add (BN or  Dropout or both) ---
class ImprovedNN(nn.Module):
    """ 
    Improved MLP, with two hidden layers, activation function ReLU.
    Add Batch Normalization and/or Dropout layers.
    """
    def __init__(self, hidden_dim1=512, hidden_dim2=256, use_bn=True, use_dropout=True,dropout_rate=0.5):
        super(ImprovedNN, self).__init__()

        #1. H1: Linear -> (BN) -> ReLU -> (Dropout)
        self.fc1=nn.Linear(INPUT_SIZE,hidden_dim1)
        self.bn1=nn.BatchNorm1d(hidden_dim1) if use_bn else nn.Identity()
        self.drop1=nn.Dropout(dropout_rate) if use_dropout else nn.Identity()

        #2. H2: Linear --> (BN) --> ReLU --> (Dropout)
        self.fc2=nn.Linear(hidden_dim1,hidden_dim2)
        self.bn2=nn.BatchNorm1d(hidden_dim2) if use_bn else nn.Identity()
        self.drop2=nn.Dropout(dropout_rate) if use_dropout else nn.Identity()

        #3. Output layer: Linear
        self.fc3=nn.Linear(hidden_dim2,NUM_CLASSES)

    def forward(self,x):
        """
        Forward process of ImprovedNN.
        Architecture: Linear -> (BN) -> ReLU -> (Dropout) -> Linear -> (BN) -> ReLU -> (Dropout) -> Linear -> Softmax (in loss function)
        """

        # H1: Linear --> (BN) --> ReLU --> (Dropout)
        x=self.drop1(F.relu(self.bn1(self.fc1(x))))

        # H2: Linear --> (BN) --> ReLU --> (Dropout)
        x=self.drop2(F.relu(self.bn2(self.fc2(x))))

        # Output layer: Linear
        x=self.fc3(x)
        return x

# --- 3. Convolutional Neural Network (CNN) ---
class BaselineCNN(nn.Module):
    """
    Simple CNN, with 2 convolutional layers + Maxpooling.
    Input shape: (batch_size, 1, 28, 28)
    """
    def __init__(self, hidden_dim_fc=128):
        super(BaselineCNN, self).__init__()
        # 1. Conv-Layer 1: Conv(1->32, 3*3) 
        # Output shape: (batch_size, 32, 28, 28)
        self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # 2. Conv-Layer 2: Conv(32->64, 3*3)
        # Output shape: (batch_size, 64, 28, 28)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # 3. Maxpooling Layer: MaxPool(2*2)
        # Output shape: (batch_size, 64, 14, 14)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

        # 4. Flatten layer and Fully Connected Layers
        self.flat_size=64*14*14  # Flatten size after conv and pooling

        # flatten -> hidden_dim_fc
        self.fc1=nn.Linear(self.flat_size, hidden_dim_fc)
        # hidden_dim_fc -> NUM_CLASSES
        self.fc2=nn.Linear(hidden_dim_fc, NUM_CLASSES)
    
    def forward(self, x):
        """ 
        Forward process of BaselineCNN.
            Architecture:
        [Conv(3*3, 32)] → ReLU → [Conv(3*3, 64)] → ReLU → MaxPool(2*2) → Flatten → Linear → ReLU → Linear → Softmax (in loss)
        """
        # Conv Layer 1 -> ReLU
        x=F.relu(self.conv1(x))

        # Conv Layer 2 -> ReLU
        x=F.relu(self.conv2(x))

        # Maxpooling
        x=self.pool(x)

        # Flatten for FC layers
        x=x.view(x.size(0), -1) # Flatten all dimensions except batch_size

        # FC Layer 1 -> ReLU
        x=F.relu(self.fc1(x))

        # Output Layer
        x=self.fc2(x)
        return x
    
# --- 4. Implement Residual Block for CNN ---
class ResidualBlock(nn.Module):
    """
    Include 2 convolutional layers with ReLU activation and Batch Normalization, and use shortcut connection.
    """
    def __init__(self,in_channels, out_channels, stride=1,use_bn=True):
        super(ResidualBlock, self).__init__()

        # First convolutional layer Conv -> (BN) -> ReLU
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # Second convolutional layer Conv -> (BN)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # Shortcut connection to match dimensions (Use 1x1 conv if in_channels != out_channels or stride != 1)
        self.shortcut=nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
            )
    
    def forward(self, x):
        """
        Forward pass of ResidualBlock.
        Architecture: x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+shortcut) -> relu -> output
        """
        identity = self.shortcut(x)  # Apply shortcut transformation if needed
        
        out = F.relu(self.bn1(self.conv1(x)))  # First conv -> BN -> ReLU
        out = self.bn2(self.conv2(out))        # Second conv -> BN
        
        out += identity  # Add shortcut connection (residual connection)
        out = F.relu(out)  # Final ReLU
        
        return out

# --- 5. Improved CNN with Residual Blocks (Support all Ablation)---
class ImprovedCNN(nn.Module):
    """
    Based on Residual Blocks to build a deeper CNN, support all ablation options.
    Architecture:
    x -> Conv(1->16,3*3) -> ReLU -> [Residual/Conv Block 1 + Downsampling] -> [Residual/Conv Block 2 + Downsampling] -> Flatten -> Dropout -> FC -> Softmax (in loss)
    """
    def __init__(self, dropout_rate=0.5, use_bn=True, use_dropout=True, use_residual=True,downsampling_type='stride_conv'):
        super(ImprovedCNN,self).__init__()

        # 1. Start conv layer
        self.conv1=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(16) if use_bn else nn.Identity()

        # 2. Define Layer 1 and Layer 2 structures
        self.use_residual=use_residual
        self.downsampling_type=downsampling_type

        # Decide block type
        self.layer1=self._make_block(16, 32, use_bn, use_residual, downsampling_type) # (32@14*14)
        self.layer2=self._make_block(32, 64, use_bn, use_residual, downsampling_type) # (64@7*7)

        # 3. Drop out
        self.drop=nn.Dropout(dropout_rate) if use_dropout else nn.Identity()

        # 4. Fully connected layers
        # Calculate flatten size based on downsampling
        if downsampling_type == 'none':
            # No downsampling: 28x28 -> 28x28 -> 28x28
            self.flat_size = 64 * 28 * 28
        else:
            # With downsampling: 28x28 -> 14x14 -> 7x7  
            self.flat_size = 64 * 7 * 7
        self.fc=nn.Linear(self.flat_size, NUM_CLASSES)

    def _make_block(self, in_channels, out_channels, use_bn, use_residual, downsampling_type):
        """
        Based on the ablation options, create a block with or without residual connections.
        """
        if use_residual:
            # Use residual blocks
            if downsampling_type=='stride_conv':
                # Default downsampling with stride in conv, use stride=2 conv for downsampling
                return ResidualBlock(in_channels, out_channels, stride=2, use_bn=use_bn)
            else:
                # Maxpool/AvgPool Ablation: Residual block remain same size(stride=1), followed by pooling
                block=ResidualBlock(in_channels, out_channels, stride=1, use_bn=use_bn)

                if downsampling_type=='maxpool':
                    pool=nn.MaxPool2d(kernel_size=2, stride=2)
                elif downsampling_type=='avgpool':
                    pool=nn.AvgPool2d(kernel_size=2, stride=2)
                elif downsampling_type=='none':
                    pool=nn.Identity() # No downsampling
                else:
                    raise ValueError(f"Unknown downsampling_type: {downsampling_type}")
                return nn.Sequential(block, pool)  
        else:
            # Use Conv -> (BN) -> ReLU to replace Residual block
            layers=[
                nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
                nn.ReLU()
            ]

            # Use Maxpooling as downsampling method
            if downsampling_type!='none':  # 修正: 'None' -> 'none'
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            return nn.Sequential(*layers)
        
    def forward(self,x):
        # Initial block
        x=F.relu(self.conv1(x)) # (B,16,28,28)

        # Layer 1 and layer 2
        x=self.layer1(x) # (B,32,14,14)
        x=self.layer2(x) # (B,64,7,7)

        # Flatten
        x=x.view(x.size(0), -1)  # Dynamic flatten: preserve batch size

        # FC layer
        x=self.drop(x)
        x=self.fc(x)

        return x




