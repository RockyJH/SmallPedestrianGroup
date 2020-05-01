"""
这个类定义了一个formation,
可以向其中添加三个相对关系的说明[[],[],[]],
get_relation（）返回的relation 经过排序
"""


class Formation(object):
    def __init__(self):
        self.ref_point_x = None  # 参照点
        self.ref_point_y = None
        self.relation = []  # 相对位置关系
        self.vn = None
        self.vc = None

    def set_vn_vc(self, vn, vc):
        self.vn = vn
        self.vc = vc

    def get_vn_vc(self):
        return self.vn, self.vc

    def set_ref_point(self, point):
        self.ref_point_x, self.ref_point_y = point

    def get_ref_point(self):
        return self.ref_point_x, self.ref_point_y

    # 添加一个list的描述
    def add_relation_positions(self, li):
        assert (len(li) == 3)
        for r in li:
            assert (len(r) == 2)
        self.relation = li

    # 逐个位置地添加
    def add_relative_position(self, r):
        if len(r) == 2:
            self.relation.append(r)
        else:
            raise Exception("向formation中添加非二元list的内容时引起失败")

    # 获取relation，返回之前，先按照从左到右的顺序排序（即relation[0]是局部坐标系中最左边的）
    # 当是river-like时，索引为[1]的元素在中间
    def get_relation(self):
        if len(self.relation) != 3:
            raise Exception("relation不是三个元组,长度是：", len(self.relation))
        else:
            return self.relation_sort(self)

    @staticmethod
    def relation_sort(self):
        self.relation = sorted(self.relation, key=lambda x: x[1])
        if self.relation[0][1] == 0 and self.relation[1][1] == 0 and self.relation[2][1] == 0:
            self.relation = sorted(self.relation, key=lambda x: x[0])[::-1]
        return self.relation
