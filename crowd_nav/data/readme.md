# 数据说明
```python
self.success_reward = 1
self.collision_penalty = -1
self.k1 = 0.08  # 速度偏向的权重
self.k2 = 0.04  # 队形差异权重
self.k3 = 3  # 到达终点判定: 距离 <=  K3 * self.agent_radius
```
有一个小组，小组外有一个agent；组外哪个Agent随机在上半圆上生成，
圆的半径是4，目标点是起点关于原点的对称点。
所有Agent的半径都是0.3,期望速度是1，
第二阶段采用orca，

```python
    rl_learning_rate = 0.001
    train_batches = 100
    train_episodes = 10000
    target_update_interval = 1000
    evaluation_interval = 1000
    capacity = 100000
    epsilon_start = 0.5
    epsilon_end = 0.1
    epsilon_decay = 4000
    checkpoint_interval = 1000
    optimizer = 'Adam'
```


