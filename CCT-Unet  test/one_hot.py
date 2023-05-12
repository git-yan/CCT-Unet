import numpy as np
from PIL import Image


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = [ ]
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)   # 沿对应轴进行与运算0，1对应行列，-1为所有维度最后一个维度主要参考shape
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


if __name__ == '__main__':
    palette = [ [ 0 ], [ 127 ], [ 255 ] ]
    mask_path = r'C:\Users\hp\Desktop\try\img\1044.png'
    mask = Image.open(mask_path)
    mask = np.array(mask)
    # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
    # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
    mask = np.expand_dims(mask, axis=2)
    mask = mask_to_onehot(mask, palette)
    print(mask.shape)
    # mask = mask.transpose([ 2, 0, 1 ])
    # mask = np.expand_dims(mask, axis=-1)
    # mask = mask.transpose([3, 0, 1, 2])
    mask = onehot_to_mask(mask, palette)
    print(mask.shape)



