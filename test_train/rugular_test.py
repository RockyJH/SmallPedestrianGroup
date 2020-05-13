import math
import matplotlib.pyplot as plt
import os
import time


class Cat:
    def __init__(self, name):
        self.name = name
        self.age = 3

    def jump(self):
        print('I am ', self.name, 'I am jumping!')
        return

    def speak(self):
        print('My name is ', self.name)


zoo = list()
for i in range(5):
    cat = Cat(str(i))
    zoo.append(cat)

ketty = Cat('ketty')
zoo.append(ketty)

for cat in zoo:
    if cat is not ketty:
        cat.jump()
    else:
        cat.speak()

# positions = []


# for i in range(-4,4):
#     positions.append([0,i])
# print(positions)
#
# s1,s2,s3 = positions[0:3]
# print ('s1,s2,s3',s1,s2,s3)


''' 测试 速度角
vx = 0
vy = 0
gx = -4
gy = 4
cx = 0
cy = 0
if vx == 0 and vy == 0:
    stop = True
    speed = 0.2
    cur_angle = math.atan2(gy - cy, gx - cx)
    print(cur_angle)
    delta_theta = math.pi / 6
    angles = [cur_angle - 2 * delta_theta, cur_angle - 1 * delta_theta, cur_angle,
              cur_angle + delta_theta, cur_angle + 2 * delta_theta]
    velocities = list()
    for angle in angles:
        velocities.append((speed * math.cos(angle), speed * math.sin(angle)))

    for i, v in enumerate(velocities):
        # p1 = [x1, y1]  # 点p1的坐标值
        # p2 = [x2, y2]  # 点p2的坐标值
        # plt.plot([x1, x2], [y1, y2])  # 简单理解就是：先写x的取值范围，再写y的取值范围
        print(v)
        p1 = [0, 0]  # 点p1的坐标值
        p2 = [v[0] * 10, v[1] * 10]  # 点p2的坐标值
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]])  # 简单理解就是：先写x的取值范围，再写y的取值范围

plt.show()
'''
