from matplotlib import animation, patches  # 画动态图
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # 基本绘图
import numpy as np

cmap = plt.cm.get_cmap('hsv', 10)  # color map


fig, ax = plt.subplots(figsize=(7, 7))  # 面板大小7，7 fig 表示一窗口 ax 是一个框
ax.tick_params(labelsize=12)  # 坐标字体大小
ax.set_xlim(-11, 11)  # -11，11  # 坐标的范围   可用于控制画面比例
ax.set_ylim(-11, 11)  # -11，11
ax.set_xlabel('x(m)', fontsize=14)
ax.set_ylabel('y(m)', fontsize=14)
show_human_start_goal = True


human_goal = mlines.Line2D([0], [4],color=cmap(2), marker='*', linestyle='None', markersize=16)
ax.add_artist(human_goal)
human_start = mlines.Line2D([0], [-4],
                            color=cmap(5),
                            marker='o', linestyle='None', markersize=4)
ax.add_artist(human_start)

robot = plt.Circle((1,2), 0.3, fill=False, color='black')
ax.add_artist(robot)

x_offset = 0.2
y_offset = 0.4
number = plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset, str(1),
                                          color='black')
ax.add_artist(number)
plt.legend([robot], ['Robot'], fontsize=14)


direction = ((1, 2), (1 + 0.3 * np.cos(np.math.pi/2), 2 + 0.3 * np.sin(np.math.pi/2)))
arrow_color = 'black'
arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)  # 箭头的长度和宽度
arrow = patches.FancyArrowPatch(*direction, color=arrow_color, arrowstyle=arrow_style)
ax.add_artist(arrow)

time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
ax.add_artist(time)
step = plt.text(0.1, 0.9, 'Step: {}'.format(0), fontsize=16, transform=ax.transAxes)
ax.add_artist(step)

plt.show()