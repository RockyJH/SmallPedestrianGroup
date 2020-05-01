# 超时、到达目标、不舒服、碰撞  等状态信息都是对象。

class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'


class Discomfort(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'Discomfort'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''