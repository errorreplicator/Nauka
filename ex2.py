import numpy as np

l0_0 = np.array([2, 0, 2, 2, 3])
l0_1 = np.array([1, 1, 0, 1, 2])
l0_2 = np.array([0, 1, 0, 2, 1])
l0_3 = np.array([1, 1, 2, 1, 1])
l0_4 = np.array([2, 0, 0, 2, 1])

l1_0 = np.array([2, 1, 1, 0, 2])
l1_1 = np.array([2, 2, 2, 0, 2])
l1_2 = np.array([2, 0, 1, 0, 2])
l1_3 = np.array([2, 1, 0, 1, 2])
l1_4 = np.array([1, 2, 0, 2, 1])

l2_0 = np.array([2, 1, 0, 1, 1])
l2_1 = np.array([2, 1, 0, 2, 0])
l2_2 = np.array([0, 2, 2, 0, 1])
l2_3 = np.array([0, 1, 2, 2, 2])
l2_4 = np.array([1, 2, 0, 1, 1])

w0_0 = [[1, -1, 1], [-1, -1, -1], [0, 0, 0]]
w0_1 = [[0, -1, -1], [-1, 1, 1], [1, 0, 0]]
w0_2 = [[-1, 1, -1], [0, -1, 1], [0, 0, -1]]

img_d0 = np.vstack([l0_0, l0_1, l0_2, l0_3, l0_4])
img_d0 = np.pad(img_d0, 1, mode='constant')
img_d1 = np.vstack([l1_0,l0_1,l0_2,l0_3,l0_4])
img_d1 = np.pad(img_d1,1,mode='constant')
img_d2 = np.vstack([l2_0,l2_1,l2_2,l2_3,l2_4])
img_d2 = np.pad(img_d2,1,mode='constant')


def r_imd2D(dim=-1):
    if dim == -1:
        return np.vstack([[img_d0],[img_d1],[img_d2]])
    else:
        return img_d0

def r_filter0(number=-1): # change name and functions to 2D and 3D
    filter_0 = np.vstack([[w0_0], [w0_1], [w0_2]])
    if number == -1:
        return  filter_0
    elif 0 <= number < 4:
        return filter_0[number]
    else:
        print('incorrect filter dimension')
        return None
