import torch


def make_anchors(feature_maps, strides, offset=0.5):
    """
    Generate anchors and corresponding stride tensors for given feature maps.

    Args:
        feature_maps (List[torch.Tensor]): List of feature maps from different levels of the network.
        strides (List[int]): List of strides corresponding to each feature map.
        offset (float, optional): Offset added to the coordinates. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - anchors: Tensor of shape (N, 2) where N is the total number of anchors across all feature maps.
            - stride_tensors: Tensor of shape (N, 1) containing the stride for each anchor.
    """
    anchors = []
    stride_tensors = []

    for feature_map, stride in zip(feature_maps, strides):
        anchors_for_map, stride_tensor = __create_anchors_for_feature_map(feature_map, stride, offset)
        anchors.append(anchors_for_map)
        stride_tensors.append(stride_tensor)

    return torch.cat(anchors), torch.cat(stride_tensors)


def __create_anchors_for_feature_map(feature_map, stride, offset):
    """
    Create anchors and stride tensor for a single feature map.

    Args:
        feature_map (torch.Tensor): A single feature map tensor.
        stride (int): The stride corresponding to this feature map.
        offset (float): Offset added to the coordinates.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - anchors: Tensor of shape (H*W, 2) where H and W are the height and width of the feature map.
            - stride_tensor: Tensor of shape (H*W, 1) filled with the stride value.
    """
    _, _, height, width = feature_map.shape
    device = feature_map.device
    dtype = feature_map.dtype

    x_coords = torch.arange(end=width, dtype=dtype, device=device) + offset
    y_coords = torch.arange(end=height, dtype=dtype, device=device) + offset

    y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
    xy_grid = torch.stack((x_grid, y_grid), dim=-1)

    anchors = xy_grid.view(-1, 2)
    stride_tensor = torch.full((height * width, 1), stride, dtype=dtype, device=device)

    return anchors, stride_tensor
