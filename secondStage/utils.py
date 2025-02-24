import nibabel as nib
import numpy as np
import cv2

def rotate(img, angle):
    """_summary_

    Args:
        img (_type_): nib.load()加载的nii数据
        angle (_type_): 自定义的旋转角度

    Returns:
        _type_: 旋转之后的nii数据
    """
    data=img.get_fdata()
    rotated_data = np.zeros_like(data)
    
    method=None
    if angle == 90:
        method=cv2.ROTATE_90_CLOCKWISE
    elif angle == 180:
        method=cv2.ROTATE_180
    elif angle == 270:
        method=cv2.ROTATE_90_COUNTERCLOCKWISE  #旋转270度相当于逆时针旋转90度
        
    if method==None:
        raise ValueError("method is None")
    
    for i in range(data.shape[2]):
        # 获取每一层
        slice_2d = data[:, :, i]
        
        # 将16x32图像拆分成两张16x16图像
        slice1 = slice_2d[:, :16]  # 左边16x16部分
        slice2 = slice_2d[:, 16:]  # 右边16x16部分
        
        # 使用cv2对每一层进行旋转
        rotated_slice1 = cv2.rotate(slice1.astype(np.uint8), method)
        rotated_slice2 = cv2.rotate(slice2.astype(np.uint8), method)
        
        # 将翻转后的两部分拼接起来，得到翻转后的16x32图像
        rotated_slice = np.concatenate((rotated_slice1, rotated_slice2), axis=1)
        
        # 将旋转后的图像重新存回到 rotated_data
        rotated_data[:, :, i] = rotated_slice

    # 创建新的NIfTI图像对象
    rotated_img = nib.Nifti1Image(rotated_data, img.affine)
    rotated_img.header['descrip'] = img.header['descrip']
    return rotated_img

def flip(img, flip_method):
    """_summary_

    Args:
        img (_type_): nib.load()加载的nii数据
        flip_method (_type_): 翻转方式，1表示水平翻转，0表示垂直翻转（两种翻转方式）

    Returns:
        _type_: 翻转之后的nii数据
    """
    data=img.get_fdata()
    flipped_data = np.zeros_like(data)

    for i in range(data.shape[2]):
        # 获取每一层
        slice_2d = data[:, :, i]
        
        # 将16x32图像拆分成两张16x16图像
        slice1 = slice_2d[:, :16]  # 左边16x16部分
        slice2 = slice_2d[:, 16:]  # 右边16x16部分

        # 分别对这两张16x16图像进行翻转
        flipped_slice1 = cv2.flip(slice1.astype(np.uint8), flip_method)
        flipped_slice2 = cv2.flip(slice2.astype(np.uint8), flip_method)

        # 将翻转后的两部分拼接起来，得到翻转后的16x32图像
        flipped_slice = np.concatenate((flipped_slice1, flipped_slice2), axis=1)

        # 将翻转后的图像重新存回到 flipped_data
        flipped_data[:, :, i] = flipped_slice

    # 创建新的NIfTI图像对象
    flipped_img = nib.Nifti1Image(flipped_data, img.affine)
    flipped_img.header['descrip'] = img.header['descrip']
    return flipped_img

def gaussianBlur(img, kernel_size):
    """_summary_

    Args:
        img (_type_): nib.load()加载的nii数据
        kernel_size (_type_): 高斯核大小

    Returns:
        _type_: 高斯模糊之后的nii数据
    """
    data=img.get_fdata()
    blurred_data = np.zeros_like(data)
    
    ksize=(kernel_size, kernel_size) # 高斯核大小
    sigmaX = 1.0     # X方向的标准差

    for i in range(data.shape[2]):
        # 获取每一层
        slice_2d = data[:, :, i]
        
        # 将16x32图像拆分成两张16x16图像
        slice1 = slice_2d[:, :16]  # 左边16x16部分
        slice2 = slice_2d[:, 16:]  # 右边16x16部分

        # 使用cv2对每一层进行高斯模糊
        blurred_slice1 = cv2.GaussianBlur(slice1.astype(np.uint8), ksize, sigmaX)
        blurred_slice2 = cv2.GaussianBlur(slice2.astype(np.uint8), ksize, sigmaX)
        
        # 将翻转后的两部分拼接起来，得到翻转后的16x32图像
        blurred_slice = np.concatenate((blurred_slice1, blurred_slice2), axis=1)
        
        # 将高斯模糊后的图像重新存回到 blurred_data
        blurred_data[:, :, i] = blurred_slice

    # 创建新的NIfTI图像对象
    blurred_img = nib.Nifti1Image(blurred_data, img.affine)
    blurred_img.header['descrip'] = img.header['descrip']
    return blurred_img