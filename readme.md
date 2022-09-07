## 使用说明

### 运行条件

python >= 3.8.0

### 文件说明

- main.py
  - 测试演示文件。

- eulerSmooth.py 
  - 使用其中的EulerSmooth类，可以方便地执行eulerSmooth操作。
- graph.py
  - 读取oco文件内容并转化为图。
- utils.py
  - 一些常用的计算函数，如距离等。
- data
  - 文件夹中储存了论文作者提供的一些oco文件。

### 快速执行说明

执行 `pip install -r requirements.txt` 下载所需文件，之后在 main.py 里将文件路径替换为恰当的文件路径，执行 main.py 即可。

### 详细执行说明

1. 执行 `pip install -r requirements.txt` 下载所需文件。
2. 在 main.py 里将文件路径替换为恰当的文件路径。
3. 执行 euler.iterater() 即为一次迭代，其中 d 代表论文中理想距离，showChange 表示是否要用箭头显示相应的变化（警告：会消耗大量的资源，建议使用简单的图片，且不要使用较小的 d 导致重采样后结点数量激增）。
4. 如需要更多配置，可以配置具体的 eulerSmooth.py 类，在 iterator() 函数里进行对应修改即可。
   - computeForces() 控制力的计算；
   - computeConstraints() 控制限制的计算；
   - moveNodes() 控制结点的移动及图的更新；
   - processFlexibleEdges() 控制迭代后处理；
   - graph.show() 会展示新的图片，实时展示新的图片。如果不需要每次都显示，可以注释掉该行。