import os
import logging

print(os.getcwd())  # 获取当前工作目录路径
print(os.path.abspath('..'))  # 获取当前工作的父目录 ！注意是父目录路径

project_path = os.path.abspath('..')
data_dir = '/crowd_nav/data-3/'
log_dir = os.path.join(project_path, 'crowd_nav/data-3/output.log')
logging.info('load form ')

print('log_dir: ', log_dir)

with open(log_dir, "r") as log_file:
    line = log_file.readline()
    print(line)
