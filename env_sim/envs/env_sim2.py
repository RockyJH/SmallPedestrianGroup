import logging  # 日志模块


import gym  # gym
import matplotlib.lines as mlines  # 基本绘图
from matplotlib import patches  # 绘制图形
import numpy as np  # 老朋友 numpy
from numpy.linalg import norm  # 线性代数

from env_sim.envs.policy.policy_factory import policy_factory  # key-value键值对
from env_sim.envs.utils.state import tensor_to_joint_state, JointState  # 状态
from env_sim.envs.utils.action import ActionRot  # 只导入了角度描述的动作还没有用到
from env_sim.envs.utils.human import Human
from env_sim.envs.utils.info import *  # 碰撞、不舒服等信息
from env_sim.envs.utils.utils import point_to_segment_dist  # 点到直线的距离

'''
对 n+1 个agent的动作仿真 其中human 被一个确定动作策略控制，
robot由一个可训练的策略控制
'''


class EnvSim(gym.Env):

    def __init__(self):

        self.time_limit = 30
        self.time_step = 0.25 # 所以最多走120步
        self.robot = None
        self.humans = None
        self.global_time = None
        self.robot_sensor_range = 5

        # reward function
        self.success_reward = 1
        self.collision_penalty = -0.25
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.5

        # simulation configuration
        self.config = None
        # self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        # self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 100, 'test': 500}
        # self.case_counter = {'train': 0, 'test': 0, 'val': 0}  # 训练，测试，验证？
        self.randomize_attributes = True  # 随机人的人的位置和速度
        self.train_val_scenario = 'circle_crossing'
        self.test_scenario = 'circle_crossing'
        self.current_scenario = None
        self.square_width = 20
        self.circle_radius = 4
        self.human_num = 5
        self.nonstop_human = False
        self.centralized_planning = False
        self.centralized_planner = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
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

    # 将初始化的工作全部放到__init__做完，configure不需要参数。
    def configure(self, config):

        self.config = config  # 后面用来为配置human

        # human_policy = 'orca'
        if self.centralized_planning:
            self.centralized_planner = policy_factory['centralized_orca']()
        # human 采用 中心化的规划 使用centralized_ORCA

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    # 机器人入场
    def set_robot(self, robot):
        self.robot = robot

    '''
    def set_group(self,group)
    
    '''
    # ----------------------------------------------

    # 按照circle_Crossing 圈圈 生成 human 实例 返回一个human实例
    # 生成human 只是站在robot的角度来生成的，因此humna 不包含目的位置信息
    def generate_human(self, human=None):
        if human is None:
            human = Human(self.config, 'humans')  # def __init__(self, config, section):
        if self.randomize_attributes:
            human.sample_random_attributes()

        # 两种场景生成human的方式不尽相同 circle_crossing 和 square_crossing
        if self.current_scenario == 'circle_crossing':
            while True:
                angle = np.random.random() * np.pi * 2  # 相当于在2pi中随即取角度
                # add some noise to simulate all the possible cases robot could meet with human
                # 为坐标添加噪声
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                # 避免随即出来的位置会直接碰撞
                # 当和其它agent的位置碰撞或和其他agent的目标位置碰撞都算碰撞，从新采样
                ###########################生成human时的一次检测碰撞#########################
                '''
                与所有robot都要检测
                '''
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + self.discomfort_dist + agent.radius
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, -px, -py, 0, 0, 0)  # 后三个是角度、半径、期望速度

        elif self.current_scenario == 'square_crossing':
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            while True:
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            while True:
                gx = np.random.random() * self.square_width * 0.5 * - sign
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                        collide = True
                        break
                if not collide:
                    break
            human.set(px, py, gx, gy, 0, 0, 0)

        return human

    def reset(self):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return: ob
        """
        if self.robot is None:  # 重置之前 必须确保robot已经设置好
            raise AttributeError('Robot has to be set!')

        self.global_time = 0  # 全局时间置零
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)

        self.current_scenario = self.test_scenario
        human_num = self.human_num
        self.humans = []
        for _ in range(human_num):
            self.humans.append(self.generate_human())

            # # case_counter is always between 0 and case_size[phase]
            # self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        # else:
        #     assert phase == 'test'
        #     if self.case_counter[phase] == -1:
        #         # for debugging purposes
        #         self.human_num = 3
        #         self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
        #         self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
        #         self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
        #         self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
        #     else:
        #         raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        # if hasattr(self.robot.policy, 'action_values'):
        #     self.action_values = list()
        # if hasattr(self.robot.policy, 'get_attention_weights'):
        #     self.attention_weights = list()
        # if hasattr(self.robot.policy, 'get_matrix_A'):
        #     self.As = list()
        # if hasattr(self.robot.policy, 'get_feat'):
        #     self.feats = list()
        # if hasattr(self.robot.policy, 'get_X'):
        #     self.Xs = list()
        # if hasattr(self.robot.policy, 'trajs'):
        #     self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = self.compute_observation_for(self.robot)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):  # action是机器人的action
        """
        为所有agent执行一步动作，返回观察、reward，done，info
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        '''第一步计算 所有humna的动作'''
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]  # agent_states 是全局的信息
            if self.robot.visible:
                agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states)[:-1]
                # 就是除去最后一行不要，最后一行是机器人的状态
            else:
                human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)  # 计算所有human的
                human_actions.append(human.act(ob))

        '''collision detection 第二步 碰撞检测'''
        dmin = float('inf')  # dmin 最小距离 初始值正无穷大
        collision = False
        # 先检测每个人和robot是否碰撞
        for i, human in enumerate(self.humans):  # 对每一个human
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # human 和robot之间的最近距离
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.info(
                    "Collision: distance between robot and p{} is {:.2E} at time {:.2E}".format(human.id, closest_dist,
                                                                                                self.global_time))
                break
            elif closest_dist < dmin:
                dmin = closest_dist  # 更新最小距离

        # collision detection between humans
        # human 之间的碰撞检测
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:  # human之间发生碰撞但是忽略不计 只是打印出日志
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))  # 执行完这一步的位置
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

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
        elif dmin < self.discomfort_dist:  # 最小距离太小了就按比例给相应的惩罚
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Discomfort(dmin)
        else:  # 其他就不奖不罚
            reward = 0
            done = False
            info = Nothing()

        if update:  # pudate = true
            # store state, action value and attention weights
            # 基本思想是将当前动作，回报等等信息存储起来，还有其他值，如果有的话
            # if hasattr(self.robot.policy, 'action_values'):
            #     self.action_values.append(self.robot.policy.action_values)
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     self.attention_weights.append(self.robot.policy.get_attention_weights())
            # if hasattr(self.robot.policy, 'get_matrix_A'):
            #     self.As.append(self.robot.policy.get_matrix_A())
            # if hasattr(self.robot.policy, 'get_feat'):
            #     self.feats.append(self.robot.policy.get_feat())
            # if hasattr(self.robot.policy, 'get_X'):
            #     self.Xs.append(self.robot.policy.get_X())
            # if hasattr(self.robot.policy, 'traj'):
            #     self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            # 这里的step 是agent的方法， 到达下一个状态
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                if self.nonstop_human and human.reached_destination():
                    self.generate_human(human)

            self.global_time += self.time_step  # 到达姆目标时间
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = self.compute_observation_for(self.robot)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info

    def compute_observation_for(self, agent):  # 计算 当前agent所观察到的状态
        if agent == self.robot:
            ob = []
            for human in self.humans:
                ob.append(human.get_observable_state())
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    # 渲染函数
    def render(self, mode='human', output_file=None):  # video
        from matplotlib import animation  # 画动态图
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.2
        y_offset = 0.4
        cmap = plt.cm.get_cmap('hsv', 10)
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
            # visualize attention scores
            # if hasattr(self.robot.policy, 'get_attention_weights'):
            #     attention_scores = [
            #         plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

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
                orientations.append(orientation) # 计算箭头的方向，也是human的朝向
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

            def update(frame_num): # frame_num 是帧数
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
                        print('you pressd the : ',event.key,' key')
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)  # matplotlib的键鼠响应事件绑定
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 500)
            anim.running = True

            '''
            1、函数FuncAnimation(fig,func,frames,init_func,interval,blit)是绘制动图的主要函数，其参数如下：
　　          a.fig 绘制动图的画布名称
　　          b.func自定义动画函数，即下边程序定义的函数update
　　          c.frames动画长度，一次循环包含的帧数，在函数运行时，其值会传递给函数update(n)的形参“n”
　　          d.init_func自定义开始帧，即传入刚定义的函数init,初始化函数
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
                logging.info("输出动图")
                anim.save("env_sim/hello.gif", writer='imagemagic', fps=12)
                #plt.show()


        else:
            raise NotImplementedError
