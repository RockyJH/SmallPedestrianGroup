# 项目说明
安装环境 pip/pip3 install -r requirements.txt
-----
从test.py 和 train.py开始.
test.py调用其他模块开始测试。共三种方法:
1. rgl_group_control.py
2. rvo_group_control.py
3. tvcg_group_control.py
三个文件在crowd_nav 目录下，该目录下data-*文件是训练得到的权重文件。
modules是 三个方法用到的其他模块。其中
> 1. explorer.py
> 2. graph_model.py
> 3. memory.py 
> 4. mlp.py 
> 5. state_predictor.py
> 6. trainer.py
> 7. value_estimator.py 
> 8. 这几个被 rgl_group_control.py 调用
> 9. box.py 被tvcg_group_control.py 调用

train.py 里 训练的是rgl_group.py,调用了所有其他的模块。env_sim包下是和环境相关的代码，config_file 包没有使用，使用的参数直接写死在程序中
envs包下env_sim.py是实验环境的主要代码，modules里包含了用到的Agent，group等类。policy.py里是一个抽象的policy类和ORCA，
unit_test包里是写代码过程中的部分测试，没有用。
