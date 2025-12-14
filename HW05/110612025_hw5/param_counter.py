import torch.nn as nn

# --- Use programmatic way to count parameters ---
def count_parameters(model: nn.Module) -> int:
    """
    Programmatically count the number of parameters in a model.
    Args:
        model (nn.Module): The neural network model.
    Outputs:
        int: Total number of parameters.
    """
    # Run through all parameters(parameters()),use numel() to get number of elements
    # Only calculate parameters that p.requires_grad == True(trainable parameters)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

# --- Use manual way to count parameters (for BaselineNN as example) ---
def manual_count_baseline_nn(layer_type: str, *args, **kwargs) -> int:
    """
    Manually count the number of parameters in BaselineNN model.
    Args:
        layer_type (str): Type of layer ('Linear' or 'Conv2d').
        *args: input and output dimensions for Linear layer.
        **kwargs: kernel size for Conv2d layer.
    Outputs:
        int: Total number of parameters.
    
    Calculation Methods:
    * Linear(in ,out): in*out + out (bias)
    * Conv2d(in_c, out_c, k_h,k_w,bias=True): out_c * (in_c * k_h * k_w) + out_c (bias)

    Noted:
        This function doesn't handle BatchNorm or Residual Block calculations.
    """

    # Convert layer_type to lowercase for easier comparison
    layer_type = layer_type.lower()

    if layer_type=='linear':
        # args: (in_features, out_features)
        in_features, out_features=args[0], args[1]
        has_bias=kwargs.get('bias',True)  # 添加這行！
        # Formula: in*out + out (bias)
        total_params=in_features*out_features+(out_features if has_bias else 0)
        return total_params

    elif layer_type=='conv2d':
        # args: (in_channels, out_channels, kernel_size)
        in_channels, out_channels, kernel_size=args[0], args[1], args[2]

        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size[0], kernel_size[1]
        
        has_bias=kwargs.get('bias',True)
        # Formula: Conv2d(in_c, out_c, k_h, k_w, bias=True)  : out_c * (in_c * k_h * k_w) + out_c (bias)
        total_params=out_channels * (in_channels * k_h * k_w) + (out_channels if has_bias else 0)
        return total_params
    # Other layer (like ReLU, Dropout, Pooling) have no parameters
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}. Only 'Linear' and 'Conv2d' are supported.")