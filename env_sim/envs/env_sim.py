import logging  # 日志模块

import gym
import matplotlib.lines as mlines  # 基本绘图
from matplotlib import patches  # 绘制图形
import numpy as np
from numpy.linalg import norm
from env_sim.envs.modules.agent import Agent
from env_sim.envs.modules.info import *
from env_sim.envs.modules.utils import point_to_segment_dist
from env_sim.envs.policy.orca import CentralizedORCA

'''
当前环境中包含一个group和若干个agent，group，对group的控制策略输入状态，返回一个formation，
以此获得组内每个agent的期望速度和目的地，然后step,对所有agent用用CentralizedORCA向下执行一步。因为只需要局部避障
'''


class EnvSim(gym.Env):

    def __init__(self):
        # 环境信息
        self.time_limit = 30
        self.time_step = 0.25
        self.out_group_agents_num = 1
        self.agent_radius = 0.3
        self.square_width = 20
        self.circle_radius = 4
        self.train_val_scenario = 'circle_crossing'
        self.test_scenario = 'circle_crossing'
        self.current_scenario = 'circle_crossing'

        self.randomize_attributes = False
        self.nonstop_human = False
        self.centralized_planner = CentralizedORCA()

        # 等待set
        self.group = None
        self.phase = None
        # 全局使用
        self.out_group_agents = list()
        self.group_actions = list()
        self.group_members = None
        self.global_time = None
        self.states = []
        self.rewards = []

        # 画轨迹
        self.want_truth = []

        # reward function ！！！！！！！！！！！！！！！
        # 在env_sim.py里用于获得真实reward，rgl_group_control.py里用于预测，一并更改！！！！！
        self.success_reward = 1
        self.collision_penalty = -1
        self.k1 = 0.08  # 速度偏向的权重
        self.k2 = 0.04  # 队形差异权重
        self.k3 = 3  # 到达终点判定: 距离 <=  K3 * self.agent_radius

        # 暂时用不到的等程序调好可删除
        self.action_values = None
        self.As = None  # Axes图表
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []
        self.test_scene_seeds = []

        self.c = 0

        logging.info('[env_sim:] out group Agent numbers: {}.'.format(self.out_group_agents_num))
        if self.randomize_attributes:
            logging.info("[env_sim:] out group Agents' prefer speed")
        else:
            logging.info("[env_sim:] out group Agents' prefer speed random?:{}。".format(self.agent_radius))
        logging.info('[env_sim:] train scene： {} , test scene: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('[env_sim:] square width: {}, circle radius: {}'.format(self.square_width, self.circle_radius))

    def set_group(self, group):
        self.group = group

    # 生成其他Agent，agent拥有一切，包括速度，指向目标的1
    def generate_agent(self):
        agent = Agent()
        if self.randomize_attributes:
            agent.sample_random_v_pref()

        if self.current_scenario == 'circle_crossing':
            while True:  # 循环直到生成一个agent实例
                angle = np.random.random() * np.pi  # 在图中上半圆
                # add some noise
                px_noise = (np.random.random() - 0.5) * agent.v_pref
                py_noise = (np.random.random() - 0.5) * agent.v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                # collide testing
                for other_agent in self.out_group_agents + self.group_members:
                    min_dist = agent.radius + other_agent.radius
                    if norm((px - other_agent.px, py - other_agent.py)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

            v_1 = -px - px
            v_2 = -py - py
            speed = np.linalg.norm(np.array((v_1, v_2)))
            agent.set(px, py, v_1 / speed, v_2 / speed, -px, -py)  # 位置，速度，目标（原点对称）

        return agent

    # 回到初始位置并返回ob
    def reset(self):
        self.global_time = 0  # 全局时间置零
        self.states.clear()
        self.rewards.clear()
        self.group_actions.clear()
        self.out_group_agents.clear()
        self.group_members = None

        self.want_truth.clear()

        # group会自动给自己一个速度指向目标的1，也是拥有一切
        # 此时会同时在环境中生成 group_members，将其添加到all_agents
        self.group.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0)
        self.group_members = self.group.get_group_members()  # group_member也是拥有一切初始属性
        self.current_scenario = self.test_scenario
        for _ in range(self.out_group_agents_num):
            agent = self.generate_agent()
            self.out_group_agents.append(agent)

        # get current observation
        ob = self.compute_observation_for(self.group)
        return ob  # group观察到的ob.

    # 返回一个observation 类型：list
    def compute_observation_for(self, cur):
        ob = []
        if cur == self.group: # 用于为group整体算ob
            for agent in self.out_group_agents:
                ob.append(agent.get_observable_state())
        else: # 为单个agent计算ob
            for other_agent in self.out_group_agents + self.group_members:
                if other_agent is not cur: # 计算其他人的ob
                    ob.append(other_agent.get_observable_state())
        return ob

    # return ob, reward, done, info
    def step(self, group_action, update=True):  # action 是group的action
        g_vx, g_vy = group_action.v
        formation = group_action.formation
        old_p = formation.get_ref_point()
        p = old_p[0] + g_vx * self.time_step, old_p[1] + g_vy * self.time_step
        relation = formation.get_relation_horizontal()  # [[a,b],[a,b],[a,b]]
        vn, vc = formation.get_vn_vc()  # vn = (x,y) vc = (x,y)

        # print(
        #     '%.1f-th' % (self.global_time / self.time_step + 1),
        #     '即将从位置:({:.2f},{:.2f}) '.format(old_p[0], old_p[1]),
        #     '采取速度:({:.2f},{:.2f}) '.format(g_vx,g_vy),
        #     '方向{:.2f}pi '.format(np.math.atan2(g_vy,g_vx)/np.math.pi),
        #     '新位置:({:.2f},{:.2f})'.format(p[0],p[1])
        # )

        # 获取新的位置,np_array的形式
        new_p1 = np.array(p) + relation[0][0] * np.array(vn) + relation[0][1] * np.array(vc)
        new_p2 = np.array(p) + relation[1][0] * np.array(vn) + relation[1][1] * np.array(vc)
        new_p3 = np.array(p) + relation[2][0] * np.array(vn) + relation[2][1] * np.array(vc)
        new_p = [new_p1, new_p2, new_p3]  # list里面是数组类型的新位置

        # 一对一分配
        old_p1 = np.array(self.group_members[0].get_position())
        old_p2 = np.array(self.group_members[1].get_position())
        old_p3 = np.array(self.group_members[2].get_position())
        old_p = [old_p1, old_p2, old_p3]

        all_orders = [[0, 1, 2], [0, 2, 1],
                      [1, 0, 2], [1, 2, 0],
                      [2, 0, 1], [2, 1, 0]]
        min_dis_index = 0
        min_dis = float('inf')

        for i, order in enumerate(all_orders):
            dis = np.linalg.norm(new_p[order[0]] - old_p[0]) \
                  + np.linalg.norm(new_p[order[1]] - old_p[1]) \
                  + np.linalg.norm(new_p[order[2]] - old_p[2])
            if dis < min_dis:
                min_dis = dis
                min_dis_index = i

        final_order = all_orders[min_dis_index]
        np1 = (new_p[final_order[0]][0], new_p[final_order[0]][1])
        np2 = (new_p[final_order[1]][0], new_p[final_order[1]][1])
        np3 = (new_p[final_order[2]][0], new_p[final_order[2]][1])

        self.group_members[0].set_goal_position(np1)
        self.group_members[1].set_goal_position(np2)
        self.group_members[2].set_goal_position(np3)

        # # 所有agents的动作---使用centralized ORCA
        # all_agents_state = [agent.get_full_state() for agent in self.group_members + self.out_group_agents]
        # all_agents_actions = self.centralized_planner.predict(all_agents_state)

        all_agents_actions = []
        for agent in (self.group_members + self.out_group_agents):
            ob = self.compute_observation_for(agent)
            action = agent.get_action(ob)
            all_agents_actions.append(action)

        collision = False
        # 检测每个group_member和组外agent是否碰撞，组外agent之间互碰不管
        for j, member in enumerate(self.group_members):
            member_action = all_agents_actions[j]
            for i, agent in enumerate(self.out_group_agents):
                px = agent.px - member.px
                py = agent.py - member.py
                vx = agent.vx - member_action.vx
                vy = agent.vy - member_action.vy
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                # member 和 组外agent之间的最近距离
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - 2 * agent.radius
                if closest_dist < 0:
                    collision = True
                    break

        # 所有agent都执行一步动作，即便碰撞，碰撞就停了
        for i, agent in enumerate(self.group_members + self.out_group_agents):
            agent.step(all_agents_actions[i])

        # group更新中心点和组速度
        self.group.update(g_vx, g_vy)

        end_position = np.array((self.group.cx, self.group.cy))  # 执行完这一步的位置
        reaching_goal = norm(end_position - np.array(self.group.get_goal())) < self.k3 * self.agent_radius

        if self.global_time >= self.time_limit:  # 在限制时间内到不了就拉到
            reward = 0
            done = True
            info = Timeout()
            logging.info("[env_sim:] time out.")
        elif collision:  # 碰撞给惩罚 gameOver
            reward = self.collision_penalty
            done = True
            info = Collision()
            logging.info("[env_sim:] collided on {} steps.".format(round(self.global_time / self.time_step) + 4))
        elif reaching_goal:  # 到达目的给奖励 gameOver
            reward = self.success_reward
            done = True
            info = ReachGoal()
            logging.info("[env_sim:]： successfully reaching the goal!")
        else:  # 其他就不奖不罚
            reward = 0
            done = False
            info = Nothing()
        # action_reward
        des_v = np.array((self.group.gx - self.group.cx, self.group.gy - self.group.cy))
        des_s = np.linalg.norm(des_v)
        des_v = des_v / des_s
        cur_velocity = (g_vx, g_vy)
        v_deviation = np.linalg.norm(np.array(cur_velocity) - des_v) / 2
        velocity_deviation_reward = self.k1 * (0.5 - v_deviation)

        cur_formation = group_action.formation
        cur_width = cur_formation.get_width()
        form_deviation = np.math.fabs(8 * self.agent_radius - cur_width) / (6 * self.agent_radius)
        form_deviation_reward = self.k2 * (0.5 - form_deviation)
        reward = reward + velocity_deviation_reward + form_deviation_reward
        # print('{:.0f}th v_d:{:.5f}  r:{:.5f}  f_d:{:.5f}  f_r:{:.5f}  all:{:.5f}'.format(
        #     self.global_time / self.time_step,
        #     v_deviation, velocity_deviation_reward,
        #     form_deviation, form_deviation_reward,
        #     reward)
        # )

        ob = None
        if update:
            self.global_time += self.time_step  # 到达目标时间
            # 每一步都把这一步所有人的状态存，前三个是 group member
            self.states.append([agent.get_full_state() for agent in self.group_members + self.out_group_agents])
            # self.rewards.append(reward)

            # 记录想去位置和orca导致的真实位置
            truth0 = self.group_members[0].get_position()
            truth1 = self.group_members[1].get_position()
            truth2 = self.group_members[2].get_position()
            self.want_truth.append([new_p1, new_p2, new_p3, truth0, truth1, truth2])

            # self.robot_actions.append(action)
            # compute the observation

            ob = self.compute_observation_for(self.group)
        return ob, reward, done, info

    def one_step_lookahead(self, action):
        return self.step(action, update=False)

    # 渲染函数，测试的时候mode使用的是video
    def render(self, mode=None, output_file=None):

        from matplotlib import animation  # 画动态图
        import matplotlib.pyplot as plt

        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 10)  # color map
        robot_color = 'black'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)  # 箭头的长度和宽度
        display_numbers = True  # 展示数字

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=15)
                ax.add_artist(human_goal)
                human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                            color=human_colors[i],
                                            marker='o', linestyle='None', markersize=15)
                ax.add_artist(human_start)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=False, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()

        if mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))  # 面板大小7，7 fig 表示一窗口 ax 是一个框
            ax.tick_params(labelsize=12)  # 坐标字体大小
            ax.set_xlim(-11, 11)  # -11，11  # 坐标的范围   可用于控制画面比例
            ax.set_ylim(-11, 11)  # -11，11
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = True

            ########画静态的轨迹##############
            for i, want_truth in enumerate(self.want_truth):
                if i % 5 == 0 or i == len(self.want_truth) - 1:
                    # want_truth === want1,want2,want3,truth1,truth2,truth3
                    for j in range(3):
                        want = plt.Circle(want_truth[j], 0.3, fill=True, color=cmap(j + 2))
                        truth = plt.Circle(want_truth[j + 3], 0.2, fill=False, color=cmap(7))
                        # number = plt.text(want_truth[j+3][0]-0.25, want_truth[j+3][1]-0.2, str(i), color='green')
                        # ax.add_artist(number)
                        ax.add_artist(want)
                        ax.add_artist(truth)

            # 用于生成图例
            circle1 = plt.Circle((1, 1), 0.3, fill=True, color=cmap(4))
            circle2 = plt.Circle((1, 1), 0.3, fill=False, color=cmap(7))

            # 在图上显示组外agent（用human标识）的起始位置和目标位置
            human_colors = [cmap(i) for i in range(len(self.out_group_agents))]
            if show_human_start_goal:  # 展示human初始目标为true时才显示
                for i in range(len(self.out_group_agents)):
                    agent = self.out_group_agents[i]
                    agent_goal = mlines.Line2D([agent.get_goal()[0]], [agent.get_goal()[1]],
                                               color=human_colors[i], marker='*', linestyle='None', markersize=8)
                    ax.add_artist(agent_goal)
                    human_start = mlines.Line2D([agent.get_start_position()[0]], [agent.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=4)
                    ax.add_artist(human_start)
            # 设置小组成员的初始位置
            for i in range(len(self.group_members)):
                agent = self.group_members[i]
                agent_start = mlines.Line2D([agent.get_start_position()[0]], [agent.get_start_position()[1]],
                                            color='black',
                                            marker='o', linestyle='None', markersize=4)
                ax.add_artist(agent_start)

            # 手动添加group的goal
            group_goal = mlines.Line2D([0], [4], color='black', marker='*', linestyle='None', markersize=16)
            ax.add_artist(group_goal)

            group_member_template = plt.Circle((0, 0), 0.3, fill=False, color='black')
            plt.legend([group_member_template, group_goal, circle1, circle2],
                       ['ROBOT', 'Goal', 'Want', 'truth'], fontsize=14)

            # 添加所有的agent
            agent_positions = [[state[j].position for j in range(self.out_group_agents_num + 3)] for state in
                               self.states]
            # group_members都是黑色
            group_members = [plt.Circle(agent_positions[0][i], self.agent_radius, fill=False, color='black') for i in
                             range(3)]
            humans = [
                plt.Circle(agent_positions[0][i], self.out_group_agents[i - 3].radius, fill=False, color=cmap(i - 3))
                for i in range(3, self.out_group_agents_num + 3)]

            all_agents = group_members + humans
            # 生成并显示数字编号
            numbers = [plt.text(all_agents[i].center[0] - x_offset, all_agents[i].center[1] + y_offset, str(i),
                                color='black') for i in range(self.out_group_agents_num + 3)]
            for i, agent in enumerate(all_agents):
                ax.add_artist(agent)
                if display_numbers:
                    ax.add_artist(numbers[i])

            # 显示时间和步骤
            time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(time)
            step = plt.text(0.1, 0.9, 'Step: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(step)

            # 计算朝向，使用箭头显示方向
            radius = self.agent_radius
            orientations = []  # [[某个成员在所有状态下的朝向],[],[],[]...]
            arrows = []
            for i in range(self.out_group_agents_num + 3):
                orientation = []
                for state in self.states:
                    agent_state = state[i]
                    theta = np.arctan2(agent_state.vy, agent_state.vx)
                    direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                    agent_state.py + radius * np.sin(theta)))
                    orientation.append(direction)
                orientations.append(orientation)  # 计算箭头的方向，也是human的朝向
                if i <= 2:
                    arrow_color = 'black'
                    arrows.append(patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style))
                else:
                    arrows.extend(
                        [patches.FancyArrowPatch(*orientation[0], color=human_colors[i - 3], arrowstyle=arrow_style)])

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):  # frame_num 是第多少帧
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num

                for i, agent in enumerate(all_agents):
                    agent.center = agent_positions[frame_num][i]
                    if display_numbers:
                        numbers[i].set_position((agent.center[0] - x_offset, agent.center[1] + y_offset))
                for arrow in arrows:
                    arrow.remove()

                arrows = []
                for i in range(self.out_group_agents_num + 3):
                    orientation = orientations[i]
                    if i <= 2:
                        arrow_color = 'black'
                        arrows.append(
                            patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color, arrowstyle=arrow_style))
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=cmap(i - 3),
                                                               arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                step.set_text('Step: {:}'.format(frame_num))

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i - 1) + ' '.join(
                        ['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    print('you pressd the : ', event.key, ' key')
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)  # matplotlib的键鼠响应事件绑定
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500)
            anim.running = True

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)
                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                anim.save("./test.gif", writer='imagemagic', fps=12)
                plt.show()

        else:
            raise NotImplementedError
