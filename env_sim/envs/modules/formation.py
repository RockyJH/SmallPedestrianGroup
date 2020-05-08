"""
Formation包含三个属性：参照点,relation,vn/vc，三个属性都必须外界显式给出
relation的数据结构使用list[[],[],[]]，因为是可变的[正方向分量,垂直方向分量]
获取relation时以局部系正方向，将三个位置描述按照横向/纵向位置关系排序，例如：
以原点为参照点，以Y轴为正方向的局部系中三点(-1,-1) (0,0) (1,1),relation可能是：
[[-1,-1],[0,0],[1,1]] 或者 [[0,0],[1,1],[-1,-1]]等6种情况，但排序后
横向排序就是 [[-1,-1],[0,0],[1,1]] 从左到右（局部系下）
纵向排序是 [[1,1],[0,0],[-1,-1]] 从上到下（局部系下）
# 测试代码：
# f = Formation()
# rel = [[0, 0], [-1, -1], [1, 1]]
# f.set_relation(rel)
# print(f.get_relation_horizontal())
# print(f.get_relation_vertical())
"""


class Formation(object):
    def __init__(self):
        self.ref_point_x = None  # 参照点
        self.ref_point_y = None
        self.relation = []  # 相对位置关系
        self.vn = None
        self.vc = None

    def set_ref_point(self, point):
        self.ref_point_x, self.ref_point_y = point

    def get_ref_point(self):
        return self.ref_point_x, self.ref_point_y

    def set_vn_vc(self, vn, vc):
        self.vn = vn
        self.vc = vc

    def get_vn_vc(self):
        return self.vn, self.vc

    # 设置relation,输入一个list,[[a,b],[c,d],[e,f]]
    def set_relation(self, li):
        assert (len(li) == 3)
        for r in li:
            assert (len(r) == 2)
        self.relation = li

    # 返回的relation按照水平方向排序
    def get_relation_horizontal(self):
        if len(self.relation) != 3:
            print('出错的relation为：', self.relation)
            raise Exception("relation不是三个元组,长度是：", len(self.relation))
        else:
            return self.sort_in_order(self, 'abreast')

    def get_width(self):
        rel = self.get_relation_horizontal()
        return (rel[2][1] - rel[0][1]) + 2 * 0.3

    def get_relation_vertical(self):
        if len(self.relation) != 3:
            raise Exception("relation不是三个元组,长度是：", len(self.relation))
        else:
            return self.sort_in_order(self, 'river')

    @staticmethod
    def sort_in_order(self, order):
        if order == 'abreast':
            self.relation = sorted(self.relation, key=lambda x: x[1])  # 按照元素的第二维升序
        elif order == 'river':
            self.relation = sorted(self.relation, key=lambda x: x[0])[::-1]  # 按照元素的第一维降序。
        return self.relation

