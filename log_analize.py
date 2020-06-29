import matplotlib.pyplot as plt
import numpy as np
import re  # 正则表达式
import os

project_path = os.getcwd()  # 项目路径
print(project_path)
data_dir = 'data-6/output.log' # 修改文件路径分析不同的log文件
log_dir = str(os.path.join(project_path, data_dir))

hor = []
ver = []

with open(log_dir, "r") as log_file:
    train_start = False
    log = log_file.read()  # 导入日志
    pattern = re.compile(r'(?<=Average loss : )[\+\-]?[\d]+[\.][\d]*?[Ee][+-]?[\d]*')  # 正则表达式匹配 Average loss : 5.01E-02
    for i, r in enumerate(re.findall(pattern, log)):
        r = float(r)
        #print('episode: {} average loss: {:.7f}'.format(i, r))
        # if i <3000:
        if i % 40 == 0:
            hor.append(i)
            ver.append(r)

    #print('end')

fig, ax = plt.subplots(figsize=(14,7))
plt.plot(hor, ver, c='r')
plt.xlabel('episode', fontsize='15', c='blue')
plt.ylabel('loss', fontsize='15', c='blue')
plt.title('loss curve', fontsize='20')
plt.show()
