import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset (for PyTorch)
    Last column in 'tensor' is assumed to be the target variable.
    """
    if tensor.shape[0] == 0:
        return 0.0
    target_col = tensor[:, -1]
    unique_classes, counts = torch.unique(target_col, return_counts=True)
    total = tensor.shape[0]
    probabilities = counts.float() / total
    entropy = 0.0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * torch.log2(prob)
    return float(entropy)

def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute (PyTorch).
    """
    if tensor.shape[0] == 0 or attribute < 0 or attribute >= tensor.shape[1] - 1:
        return 0.0
    attribute_column = tensor[:, attribute]
    total = tensor.shape[0]
    unique_values = torch.unique(attribute_column)
    avg_info = 0.0
    for val in unique_values:
        mask = (attribute_column == val)
        subset = tensor[mask]
        weight = subset.shape[0] / total
        if subset.shape[0] > 0:
            subset_entropy = get_entropy_of_dataset(subset)
            avg_info += weight * subset_entropy
    return float(avg_info)

def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate information gain for an attribute in the dataset (PyTorch).
    """
    if tensor.shape[0] == 0:
        return 0.0
    ds_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = ds_entropy - avg_info
    return round(float(info_gain), 4)

def get_selected_attribute(tensor: torch.Tensor):
    """
    Return:
      - dict: {attribute_index: information_gain}
      - int: index of attribute with highest info gain
    for PyTorch tensors.
    """
    if tensor.shape[0] == 0 or tensor.shape[1] <= 1:
        return {}, -1
    num_attributes = tensor.shape[1] - 1
    gain_dict = {}
    for i in range(num_attributes):
        gain_dict[i] = get_information_gain(tensor, i)
    if not gain_dict:
        return {}, -1
    selected_attr = max(gain_dict, key=gain_dict.get)
    return gain_dict, selected_attr
