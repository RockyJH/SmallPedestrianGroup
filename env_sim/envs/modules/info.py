# 超时、到达目标、不舒服、碰撞  等状态信息都是对象。


class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return '超过时间限度而未到达！'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return '到达目的地！'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return '发生碰撞！'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''
