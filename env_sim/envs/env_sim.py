import logging  # 日志模块
import random  # 有random() 方法  返回随机生成的一个实数，它在[0,1)范围内。
import math  # 数学函数，一般是对c库中同名函数的简单封装

import gym
import matplotlib.lines as mlines  # 基本绘图
from matplotlib import patches  # 绘制图形
import numpy as np
from numpy.linalg import norm
from env_sim.envs.utils.agent import Agent
from env_sim.envs.utils.state import tensor_to_joint_state, JointState  # 状态
from env_sim.envs.utils.info import *
from env_sim.envs.utils.utils import point_to_segment_dist
from env_sim.envs.policy.orca import CentralizedORCA

'''
当前环境中包含一个group和若干个agent，group，对group的控制策略输入状态，返回一个formation，
以此获得组内每个agent的期望速度和目的地，然后step,对所有agent用用CentralizedORCA向下执行一步。因为只需要局部避障
'''


class EnvSim(gym.Env):

    def __init__(self):

        self.group_actions = None
        self.time_limit = None
        self.time_step = None
        self.group = None
        self.out_group_agents = None
        self.group_members = None
        self.out_group_agents_num = None
        self.global_time = None
        self.robot_sensor_range = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.formation_value = None

        # simulation configuration
        self.config = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.nonstop_human = None

        #
        self.states = None
        self.action_values = None
        self.robot_actions = None
        self.rewards = None

        self.As = None  # Axes图表
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []
        self.phase = None
        self.centralized_planner = CentralizedORCA
        self.agent_radius = 0.3

    def configure(self, config):

        self.config = config  # 后面用来为配置human

        logging.info('agent number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_group(self, group):
        self.group = group

    # 生成其他Agent
    def generate_agent(self):
        agent = Agent(self.config, 'agent')
        if self.randomize_attributes:
            agent.sample_random_attributes()

        if self.current_scenario == 'circle_crossing':
            while True:  # 一直循环直到成功的生成一个agent实例
                angle = np.random.random() * 2 * np.pi  # 在2pi中随机取角度
                # add some noise
                px_noise = (np.random.random() - 0.5) * agent.v_pref
                py_noise = (np.random.random() - 0.5) * agent.v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                # collide testing
                for other_agent in self.all_agents:
                    min_dist = agent.radius + other_agent.radius
                    if norm((px - other_agent.px, py - other_agent.py)) < min_dist or \
                            norm((px - other_agent.gx, py - other_agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            agent.set(px, py, -px, -py, 0, 0, 0)  # 后三个是、半径、期望速度
            self.out_group_agents.append(agent)

        return agent

    # 回到初始位置并返回ob
    def reset(self):
        self.global_time = 0  # 全局时间置零
        self.group.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0)
        # 此时会同时在环境中生成 group_members，将其添加到all_agents
        self.group_members = self.group.get_group_members()

        self.current_scenario = self.test_scenario
        other_agent_num = self.out_group_agents_num
        for _ in range(other_agent_num):
            self.generate_agent()

        # 让所有地方的agent同步起来
        for agent in self.group_members + self.out_group_agents:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        self.rewards = list()
        self.group_actions = list()

        # get current observation
        ob = self.compute_observation_for(self.group)

        return ob  # group观察到的ob.

    def one_step_lookahead(self, action):
        return self.step(action, update=False)

    # return ob, reward, done, info
    def step(self, action, update=True):  # action 是group的action
        # 为所有agent计算出一步动作，然后碰撞检查，更新环境，返回(ob, reward, done, info)

        # 先处理group_members,主要是set_goal(self, gx, gy)
        # 速度不用设置，但是执行完环境更新应该更新group_members的速度

        g_vx, g_vy = action[0]
        formation = action[1]
        p = formation.get_ref_point()
        p = p[0] + g_vx * self.time_step, p[1] * g_vy * self.time_step
        relation = formation.get_relation()  # [[a,b],[a,b],[a,b]]
        vn, vc = formation.get_vn_vc()  # vn = (x,y) vc = (x,y)

        # 获取新的位置,np_array的形式
        new_p1 = np.array(p) + relation[0][0] * np.array(vn) + relation[0][1] * np.array(vc)
        new_p2 = np.array(p) + relation[1][0] * np.array(vn) + relation[1][1] * np.array(vc)
        new_p3 = np.array(p) + relation[2][0] * np.array(vn) + relation[2][1] * np.array(vc)
        new_p_array = [new_p1, new_p2, new_p3]  # list里面是数组类型的新位置

        # 一对一地分配谁去哪里

        old_p1 = np.array(self.group_members[0].get_position())
        old_p2 = np.array(self.group_members[1].get_position())
        old_p3 = np.array(self.group_members[2].get_position())

        old_p_arr = [old_p1, old_p2, old_p3]

        all_premutations = [[1, 2, 3], [1, 3, 2],
                            [2, 1, 3], [2, 3, 2],
                            [3, 1, 2], [3, 2, 1]]
        min_dis_index = 0
        min_dis = float('inf')

        for i, premutation in enumerate(all_premutations):
            dis = np.linalg.norm(new_p_array[premutation[0]] - old_p_arr[0] +
                                 new_p_array[premutation[1]] - old_p_arr[1] +
                                 new_p_array[premutation[2]] - old_p_arr[2])
            if dis < min_dis:
                min_dis = dis
                min_dis_index = i

        final_premutation = all_premutations[min_dis_index]
        self.group_members[0].set_goal(new_p_array[final_premutation[0]][0], new_p_array[final_premutation[0]][1])
        self.group_members[1].set_goal(new_p_array[final_premutation[1]][0], new_p_array[final_premutation[1]][1])
        self.group_members[2].set_goal(new_p_array[final_premutation[2]][0], new_p_array[final_premutation[2]][1])

        # 所有agents的动作
        agent_states = [agent.get_full_state() for agent in self.group_members + self.out_group_agents]
        agent_actions = self.centralized_planner.predict(agent_states)[:-1]

        dmin = float('inf')  # dmin 最小距离 初始值正无穷大
        collision = False
        # 检测每个group_member和组外agent是否碰撞，组外agent之间互碰不管
        for j, member in enumerate(self.group_members):
            member_action = agent_actions[j]
            for i, agent in enumerate(self.out_group_agents):
                px = agent.px - member.px
                py = agent.py - member.py
                vx = agent.vx - member_action.vx
                vy = agent.vy - member_action.vy
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                # member 和 组外agent之间的最近距离
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - agent.radius
                if closest_dist < 0:
                    collision = True
                    logging.info(
                        "Collision happend at time {:.2E}".format(self.global_time))
                    break

        # check if reaching the goal
        # 让所有group_member执行动作，来更新group
        for i, member in self.group_members:
            member.step(agent_actions[i])

        self.group.update()

        end_position = np.array((self.group.px, self.group.py))  # 执行完这一步的位置
        reaching_goal = norm(end_position - np.array(self.group.get_goal_position())) < 3 * self.agent_radius

        if self.global_time >= self.time_limit - 1:  # 在限制时间内到不了就拉到
            reward = 0
            done = True
            info = Timeout()
        elif collision:  # 碰撞给惩罚 gameOver
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:  # 到达目的给奖励 gameOver
            reward = self.success_reward
            done = True
            info = ReachGoal()
        else:  # 其他就不奖不罚
            reward = 0
            done = False
            info = Nothing()
        # formation_reward

        if update:
            # update all agents
            for agent, action in zip(self.out_group_agents, agent_actions[3:]):
                agent.step(action)
                # if self.nonstop_human and human.reached_destination():
                #     self.generate_human(human)

            self.global_time += self.time_step  # 到达姆目标时间
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            ob = self.compute_observation_for(self.group)

        return ob, reward, done, info

    def compute_observation_for(self, group):  # 计算 当前agent所观察到的状态
        ob = []
        for agent in self.out_group_agents:
            ob.append(agent.get_observable_state())
        return ob

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
            fig, ax = plt.subplots(figsize=(7, 7))  # 7，7
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
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))  # 面板大小7，7 fig 表示一窗口 ax 是一个框
            ax.tick_params(labelsize=12)  # 坐标字体大小
            ax.set_xlim(-11, 11)  # -11，11  # 坐标的范围   可用于控制画面比例
            ax.set_ylim(-11, 11)  # -11，11
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = True

            # 在图上显示human的起始位置和目标位置
            human_colors = [cmap(i) for i in range(len(self.humans))]
            if show_human_start_goal:  # 展示human初始目标为true时才显示
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i], marker='*', linestyle='None', markersize=8)
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=4)
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=8)
            ax.add_artist(robot_start)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color=robot_color, marker='*', linestyle='None',
                                 markersize=15, label='Goal')

            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=14)

            # add humans and their numbers

            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=cmap(i))
                      for i in range(len(self.humans))]
            # len(self.humans) == 5
            # 是否展示human的数字
            if display_numbers:
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i),
                                          color='black') for i in range(len(self.humans))]

            for i, human in enumerate(humans):
                ax.add_artist(human)
                if display_numbers:
                    ax.add_artist(human_numbers[i])

            # add time annotation 显示时间注释
            time = plt.text(0.4, 0.9, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(time)
            step = plt.text(0.1, 0.9, 'Step: {}'.format(0), fontsize=16, transform=ax.transAxes)
            ax.add_artist(step)

            # 计算朝向，使用箭头显示方向 compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            orientations = []
            for i in range(self.human_num + 1):
                orientation = []
                for state in self.states:
                    agent_state = state[0] if i == 0 else state[1][i - 1]
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(agent_state.theta),
                                                               agent_state.py + radius * np.sin(agent_state.theta)))
                    else:
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        direction = ((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                                                        agent_state.py + radius * np.sin(theta)))
                    orientation.append(direction)
                orientations.append(orientation)  # 计算箭头的方向，也是human的朝向
                if i == 0:
                    arrow_color = 'black'
                    arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)]
                else:
                    arrows.extend(
                        [patches.FancyArrowPatch(*orientation[0], color=human_colors[i - 1], arrowstyle=arrow_style)])

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            if len(self.trajs) != 0:
                human_future_positions = []
                human_future_circles = []
                for traj in self.trajs:
                    human_future_position = [[tensor_to_joint_state(traj[step + 1][0]).human_states[i].position
                                              for step in range(self.robot.policy.planning_depth)]
                                             for i in range(self.human_num)]
                    human_future_positions.append(human_future_position)

                for i in range(self.human_num):
                    circles = []
                    for j in range(self.robot.policy.planning_depth):
                        circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius / (1.7 + j),
                                            fill=False, color=cmap(i))
                        ax.add_artist(circle)
                        circles.append(circle)
                    human_future_circles.append(circles)

            def update(frame_num):  # frame_num 是帧数
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]

                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    if display_numbers:
                        human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] + y_offset))
                for arrow in arrows:
                    arrow.remove()

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color='black',
                                                          arrowstyle=arrow_style)]
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=cmap(i - 1),
                                                               arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                    # if hasattr(self.robot.policy, 'get_attention_weights'):
                    #     attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                step.set_text('Step: {:}'.format(frame_num))

                if len(self.trajs) != 0:
                    for i, circles in enumerate(human_future_circles):
                        for j, circle in enumerate(circles):
                            circle.center = human_future_positions[global_step][i][j]

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
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                    if event.key == 'n':
                        print('you pressd the : ', event.key, ' key')
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)  # matplotlib的键鼠响应事件绑定
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500)
            anim.running = True

            '''
        1、函数FuncAnimation(fig, func, frames, init_func, interval, blit)
        是绘制动图的主要函数，其参数如下：

　　          a.fig
绘制动图的画布名称
　　          b.func自定义动画函数，即下边程序定义的函数update
　　          c.frames动画长度，一次循环包含的帧数，在函数运行时，其值会传递给函数update(n)
的形参“n”
　　          d.init_func自定义开始帧，即传入刚定义的函数init, 初始化函数
　　          e.interval更新频率，以ms计
　　          f.blit选择更新所有点，还是仅更新产生变化的点。应选择True，但mac用户请选择False，否则无法显
'''


            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                anim.save("env_sim/hello.gif", writer='imagemagic', fps=12)
                plt.show()

        else:
            raise NotImplementedError
