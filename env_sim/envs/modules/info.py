# 超时、到达目标、不舒服、碰撞  等状态信息都是对象。


class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Time out!'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reached the goal!'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collided!'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''
