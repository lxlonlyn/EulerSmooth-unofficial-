r"""
    描述：
        测试程序。
"""
import matplotlib.pyplot as plt
from graph import *
from eulerSmooth import *


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    euler = EulerSmooth(ExtendedGraph(InitialGraph('data/Imdb20.oco'))) # 更改文件路径以查看不同图的效果
    euler.graph.show()
    for i in range(300):
        logging.info("iterate times: " + str(i + 1))
        plt.ion()
        plt.cla()
        euler.iterater(d=10, showChange=False)
        plt.show()
        plt.pause(0.1)
        plt.ioff()
    plt.show()
