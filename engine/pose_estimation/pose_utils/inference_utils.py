import os
import torch

from pose_utils.camera import get_focalLength_from_fieldOfView


# | 参数名          | 含义                            |
# | ------------ | ----------------------------- |
# | `img_size`   | 图像的宽度或高度（假设正方形）               |
# | `fov`        | 视场角 Field of View（单位：度）       |
# | `p_x`, `p_y` | 主点坐标（归一化 0\~1），若为 None 默认图像中心 |
# | `device`     | 返回的相机矩阵所在的计算设备（如 GPU）         |
# 返回，3*3相机内参
def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device("cuda")):
    """Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
        K[0, -1], K[1, -1] = p_x * img_size, p_y * img_size
    else:
        K[0, -1], K[1, -1] = img_size // 2, img_size // 2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K
