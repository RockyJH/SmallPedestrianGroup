import numpy as np

'''
点到直线的距离
'''
def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    计算点(x3,y3) 到直线(x1,y1)----(x2,y2)
    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:  # 直线变成一个点
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))
