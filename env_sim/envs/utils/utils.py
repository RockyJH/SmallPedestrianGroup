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
        return np.linalg.norm((x3 - x1, y3 - y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y - y3))


# 由一个速度计算vn,vc
# vn:沿着速度方向的单位向量
# vc:与vn成右手系的单位向量
def compute_vn_vc(vx, vy):
    tem = np.math.hypot(vx, vy)  # 根号下x^2+y^2
    vector_n = (vx / tem, vy / tem)  # 得局部坐标系的y方向单位向量(在全局坐标的描述)
    vector_nc = (vector_n[1], -vector_n[0])  # 局部坐标系下x方向单位向量
    return vector_n, vector_nc


def normalize_vector(vx,vy):
    return np.linalg.norm(np.array((vx,vy)))
