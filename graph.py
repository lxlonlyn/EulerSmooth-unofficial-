r"""
    文件描述：
        读入、存储、绘制图。
"""
import copy
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Any, Union


class Subgraph:
    """
        描述：
            子图，记录初始图的子图信息。
        
        属性：
            - nodesID(list[str]): 所含结点编号
            - edgesID(list[str]): 所含边编号
            - attribute(dict): 其他属性信息，包括颜色等
    """
    nodesID: List[str] = []
    edgesID: List[str] = []
    attribute: dict = {}

    def __init__(self) -> None:
        r"""
            描述：
                初始化建图。
            参数：无
        """
        self.nodesID = []
        self.edgesID = []
        self.attribute = {}

class InitialGraph:
    r"""
        描述：
            初始图。如果需要展示图片或进行操作，需要转化为扩展图。

        属性：
            - nodes(dict): 结点，格式为 结点编号->{结点信息}
            - edges(dict): 边（弧），格式为 边编号->{边信息}
            - subgraphs(list[Subgraph]): 子图，子图无具体编号区分，通过下标选择
            - numberOfSubgraphs(int): 目前的子图总数
    """
    nodes: dict = {} 
    edges: dict = {} 
    subgraphs: List[Subgraph] = []
    numberOfSubgraphs: int = 0 

    def __init__(self, filePath: str = "") -> None:
        r"""
            描述：
                初始化建图，根据文件路径构造原始图。
            参数：
                - filePath(str): oco文件路径
        """
        self.nodes = {}
        self.edges = {}
        self.subgraphs = []
        self.numberOfSubgraphs = 0

        # 检查文件是否符合格式，或者仅需初始化而不用赋值
        if not filePath:
            return
        if not filePath.endswith('.oco'):
            logging.warning("警告：仅支持oco文件格式，图加载失败")
            return 
        fileLines = None
        with open(filePath, "r", encoding='utf') as f:
            fileLines = f.readlines()
        if not fileLines:
            logging.warning("警告：文件加载错误或文件内容为空")
            return 
        
        # 解析文件内容
        attrs = []
        defaultValue = []
        dataType = ''
        totLinesRead = 0
        for line in fileLines:
            totLinesRead += 1
            line = line.strip()
            isDataLine = False
            if not len(line): 
                continue
            elif line.startswith("#graph"):
                continue
            elif line.startswith("##graph"):
                isDataLine = True
                dataType = 'subgraph'
                self.numberOfSubgraphs += 1
                self.subgraphs.append(Subgraph())
            elif line.startswith('#nodes'):
                isDataLine = True
                dataType = 'nodes'
            elif line.startswith('#edges'):
                isDataLine = True
                dataType = 'edges'
            elif line.startswith("@attribute"):
                isDataLine = True
                attrs = line.split()
                for i in range(len(attrs)):
                    if attrs[i].startswith("@"):
                        attrs[i] = attrs[i][1:]
            elif line.startswith("@type"):
                isDataLine = True
            elif line.startswith("@default"):
                isDataLine = True
                defaultValue = line.split()
                if dataType == 'subgraph':
                    for i in range(1, len(defaultValue)):
                        self.subgraphs[self.numberOfSubgraphs - 1].attribute.update({attrs[i]: defaultValue[i]})
            else:
                isDataLine = True
                data = line.split('\t')
                ele = {}
                if self.numberOfSubgraphs:
                    if dataType == 'nodes':
                        self.subgraphs[self.numberOfSubgraphs - 1].nodesID.append(data[0])
                    elif dataType == 'edges':
                        self.subgraphs[self.numberOfSubgraphs - 1].edgesID.append(data[0])
                    else:
                        logging.warning(str(totLinesRead) + " 行警告：文件格式错误，未指定数据类型")
                else:
                    for i in range(1, len(data)):
                        if attrs[i] == 'nodePosition' or attrs[i] == 'nodeSize':
                            data[i] = (float(data[i].strip().split(',')[0]), float(data[i].strip().split(',')[1]))
                            data[i] = np.array(data[i], dtype=float)
                        elif attrs[i] == 'edgePoints':
                            resData = []
                            data[i] = data[i].split()
                            for position in data[i]:
                                if len(position) < 3: 
                                    continue
                                resData.append(np.array([float(position.strip().split(',')[0]), float(position.strip().split(',')[1])], dtype=float))
                            data[i] = resData
                        ele[attrs[i]] = data[i]
                    if dataType == 'nodes':
                        self.nodes[data[0]] = ele
                    elif dataType == 'edges':
                        self.edges[data[0]] = ele
                    else:
                        logging.warning(str(totLinesRead) + " 行警告：文件格式错误，未指定数据类型")
            if not isDataLine:
                dataType = ''

        # 如果没有子图，则整体算作一个子图
        if not len(self.subgraphs):
            self.numberOfSubgraphs += 1
            self.subgraphs.append(Subgraph())
            self.subgraphs[-1].nodesID = list(self.nodes.keys())
            self.subgraphs[-1].edgesID = list(self.edges.keys())

class Node:
    r"""
        描述：
            结点类，储存结点的基本信息。

        属性：
            - position(number|(number,number)): 结点位置
            - degree(int): 结点度数
            - connectedEdges(set(str)): 相连边编号
            - size(number|(number,number)): 结点大小
            - attribute(dict): 其他属性
    """
    position: np.array = np.zeros(2, dtype=float)
    degree: int = 0
    connectedEdges: Set[str] = set()
    size: np.array = np.ones(2, dtype=float)
    attribute: Dict[str, Any] = {}

    def __init__(self, _position = (0, 0), _degree = 0, _connectedEdges = set(), _size = (1, 1), _attribute = {}) -> None:
        """
            描述：
                初始化结点构造。

            参数：
                - _position(ndarray=[float,float]): 结点位置
                - _degree(int): 结点度数
                - _connectedEdges(set(str)): 相连边
                - _size(ndarray=[float,float]): 结点大小
                - _attribute(dict): 其他属性
        """
        try:
            self.position = np.array([_position[0], _position[1]], dtype=float)
        except Exception:
            logging.warning("警告：初始化结点位置错误，将使用(0, 0)替代")
            self.position = np.zeros(2, dtype=float)
        self.degree = _degree
        self.connectedEdges = copy.deepcopy(_connectedEdges)
        if isinstance(_size, (int, float)):
            self.size = np.array([_size, _size], dtype=float)
        else:
            try:
                self.size = np.array([_size[0], _size[1]], dtype=float)
            except Exception:
                logging.warning("警告：初始化结点大小错误，将使用(1, 1)替代")
                self.size = np.ones(2, dtype=float)
        self.attribute = copy.deepcopy(_attribute)
        
class Edge:
    r"""
        描述：
            线段类，储存线段的基本信息。

        属性：
            - extremities((str,str)): 端点编号
    """
    extremities: Tuple[str, str] = ("", "")

    def __init__(self, _extremities: Union[Tuple[str, str], List[str]]) -> None:
        r"""
            描述：初始化建边。

            参数：
                - _extremities((str,str)): 端点编号

        """
        self.extremities = (_extremities[0], _extremities[1])

    def getAnotherExtremity(self, extremity: str) -> str:
        r"""
            描述：已知一个端点编号，查找另一个端点编号。

            参数：
                - extremity: 已知端点编号
        """
        if extremity == self.extremities[0]:
            return self.extremities[1]
        elif extremity == self.extremities[1]:
            return self.extremities[0]
        else:
            logging.warning("Edge.getAnotherExtremity(): 已知端点不在两端点中: " + extremity)
            return ""

    def getCommonExtremity(self, edge: 'Edge') -> str:
        r"""
            描述：
                查找两者共同的端点。

            参数：
                - edge(Edge): 另一个边
        """
        if self.extremities[0] == edge.extremities[0] or self.extremities[0] == edge.extremities[1]:
            return self.extremities[0]
        elif self.extremities[1] == edge.extremities[0] or self.extremities[1] == edge.extremities[1]:
            return self.extremities[1]
        else:
            logging.warning("警告：没有相同结点" + str(self.extremities) + " " + str(edge.extremities))
            return ""

class ExtendedSubgraph:
    r"""
        描述：
            扩展子图。记录扩展图的子图信息。

        属性：
            - nodesID(list[str]): 子图内结点编号
            - edgesID(list[str]): 子图内线段编号，需要保证线段首尾相接为一个环
            - attribute(dict): 子图其他属性信息，包括颜色等
    """
    nodesID: List[str] = []
    edgesID: List[str] = []
    attribute: dict = {}

    def __init__(self) -> None:
        """
            描述：
                初始化建图。

            参数: 无
        """
        self.nodesID = []
        self.edgesID = []
        self.attribute = {}

class ExtendedGraph:
    r"""
        描述：
            扩展图。在初始图的基础上进行了一系列扩展，方便进行相关计算、修改及绘图。

        属性：
            - nodes(dict): 结点，格式为 结点编号->{结点信息}，边上结点包含在内
            - edges(dict): 线段，格式为 线段编号->{线段信息}，线段由初始图边拆分而来
            - nodeIndex(int): 目前最大的结点编号
            - edgeIndex(int): 目前最大的线段编号
            - subgraphs(list[ExtendedSubgraph]): 子图
    """
    nodes: Dict[str, Node] = {}
    edges: Dict[str, Edge] = {}
    nodeIndex: int = 0
    edgeIndex: int = 0
    subgraphs: List[ExtendedSubgraph] = []

    def __init__(self, g: InitialGraph) -> None:
        r"""
            描述：
                根据初始图构建扩展图。
                
            参数：
                g(InitialGraph): 初始图
        """
        # 加入初始图结点
        for nodeID in g.nodes:
            self.nodes[nodeID] = Node(g.nodes[nodeID]['nodePosition'])
        self.nodeIndex = 0
        self.edgeIndex = 0
        # 计算初始结点的度数
        connectedCurve: Dict[str, List[str]] = {}
        for edgeID in g.edges:
            u = g.edges[edgeID]['from'] 
            v = g.edges[edgeID]['to']
            self.nodes[u].degree += 1
            self.nodes[v].degree += 1
            if u not in connectedCurve:
                connectedCurve[u] = []
            if v not in connectedCurve:
                connectedCurve[v] = []
            connectedCurve[u].append(edgeID)
            connectedCurve[v].append(edgeID)
        positionToNode = {}
        extremitiesToEdge = {}
        # 计算子图的边缘路径
        for subgraph in g.subgraphs:
            # 获取子图的弧路径
            s = g.edges[subgraph.edgesID[0]]['from']
            curvePath = [(subgraph.edgesID[0], 0)]
            u = s
            v = g.edges[subgraph.edgesID[0]]['to']
            while not v == s:
                err = False
                u = v
                for nextEdgeID in connectedCurve[u]:
                    if (nextEdgeID, 0) in curvePath or (nextEdgeID, 1) in curvePath:
                        continue
                    if nextEdgeID not in subgraph.edgesID:
                        continue
                    nextEdge = g.edges[nextEdgeID]
                    v = nextEdge['to'] if (u == nextEdge['from']) else nextEdge['from']
                    if v in subgraph.nodesID:
                        if v == nextEdge['to']:
                            curvePath.append((nextEdgeID, 0))
                        else:
                            curvePath.append((nextEdgeID, 1))
                    else:
                        logging.warning("警告：子图中边指向不在子图中的点")
                        err = True
                    break
                if err:
                    break
            if not v == s:
                logging.warning("警告：终点不是起点: " + v + " " + s)
            # 边缘弧路径扩展为边缘点路径
            nodePath = []
            for (curveID, order) in curvePath:
                if order == 0:
                    # 顺序
                    nodePath.append(g.edges[curveID]['from'])
                    for bendPosition in g.edges[curveID]['edgePoints']:
                        if tuple(bendPosition) not in positionToNode:
                            positionToNode[tuple(bendPosition)] = self.addNode(bendPosition)
                        nodePath.append(positionToNode[tuple(bendPosition)])
                else:
                    # 逆序
                    nodePath.append(g.edges[curveID]['to'])
                    for bendPosition in reversed(g.edges[curveID]['edgePoints']):
                        if tuple(bendPosition) not in positionToNode:
                            positionToNode[tuple(bendPosition)] = self.addNode(bendPosition)
                        nodePath.append(positionToNode[tuple(bendPosition)])
            # 边缘点路径扩展为边缘线段路径
            edgePath = []
            for i in range(1, len(nodePath)):
                if (nodePath[i - 1], nodePath[i]) in extremitiesToEdge:
                    edgeID = extremitiesToEdge[(nodePath[i - 1], nodePath[i])]
                elif (nodePath[i], nodePath[i - 1]) in extremitiesToEdge:
                    edgeID = extremitiesToEdge[(nodePath[i], nodePath[i - 1])]
                else:
                    edgeID = self.addEdge(nodePath[i - 1], nodePath[i])
                    extremitiesToEdge[(nodePath[i - 1], nodePath[i])] = edgeID
                edgePath.append(edgeID)
            if len(nodePath) > 1:
                if (nodePath[-1], nodePath[0]) in extremitiesToEdge:
                    edgeID = extremitiesToEdge[(nodePath[-1], nodePath[0])]
                elif (nodePath[0], nodePath[-1]) in extremitiesToEdge:
                    edgeID = extremitiesToEdge[(nodePath[0], nodePath[-1])]
                else:
                    edgeID = self.addEdge(nodePath[-1], nodePath[0])
                    extremitiesToEdge[(nodePath[-1], nodePath[0])] = edgeID
                edgePath.append(edgeID)
            # 加入子图中非边缘结点
            for nodeID in subgraph.nodesID:
                if nodeID not in nodePath:
                    nodePath.append(nodeID)
            # 加入子图数据
            newSubgraph = ExtendedSubgraph()
            newSubgraph.nodesID = nodePath
            newSubgraph.edgesID = edgePath
            newSubgraph.attribute = copy.deepcopy(subgraph.attribute)
            self.subgraphs.append(newSubgraph)
        # 之前度数包括了弧的度数，需要减去
        for edgeID in g.edges:
            u = g.edges[edgeID]['from'] 
            v = g.edges[edgeID]['to']
            self.nodes[u].degree -= 1
            self.nodes[v].degree -= 1

    def addNode(self, newNodePosition) -> str:
        """
            描述：
                新增结点。
            
            参数：
                - newNodePosition: 新结点的坐标

            返回值：
                newNodeID(str): 新结点的编号
        """
        while (self.nodeIndex == 0) or (str(self.nodeIndex) + "n" in self.nodes.keys()):
            self.nodeIndex += 1
        newNodeID = str(self.nodeIndex) + "n"
        self.nodes[newNodeID] = Node((newNodePosition[0], newNodePosition[1]))
        return newNodeID
    
    def addEdge(self, newEdgeFrom: str, newEdgeTo: str) -> str:
        """
            描述：
                新增线段。
            
            参数：
                - newEdgeFrom(str): 第一个线段端点的编号
                - newEdgeTo(str): 第二个线段端点的编号

            返回值：
                newNodeID(str): 新线段的编号
        """
        while (self.edgeIndex == 0) or (str(self.edgeIndex) + "e" in self.edges.keys()):
            self.edgeIndex += 1
        newEdgeID = str(self.edgeIndex) + "e"
        self.edges[newEdgeID] = Edge((newEdgeFrom, newEdgeTo))
        self.nodes[newEdgeFrom].degree += 1
        self.nodes[newEdgeTo].degree += 1
        self.nodes[newEdgeFrom].connectedEdges.add(newEdgeID)
        self.nodes[newEdgeTo].connectedEdges.add(newEdgeID)
        return newEdgeID

    def show(self) -> None:
        """
            描述：
                绘制目前图形。

            参数：无
        """
        # 绘制边缘
        for subgraph in self.subgraphs:
            currentColor = None
            if 'color' in subgraph.attribute:
                currentColor = subgraph.attribute['color']
            position = []
            for edgeID in subgraph.edgesID:
                edge = self.edges[edgeID]
                if len(position) and (self.nodes[edge.extremities[0]].position == position[-1]).all():
                    position.append(self.nodes[edge.extremities[1]].position)
                else:
                    position.append(self.nodes[edge.extremities[0]].position)
            if currentColor:
                plt.fill([e[0] for e in position], [e[1] for e in position], color=currentColor, alpha=0.5)
            else:
                plt.fill([e[0] for e in position], [e[1] for e in position], alpha=0.5)
        # 绘制集合点
        for nodeID in self.nodes:
            if self.nodes[nodeID].degree == 0:
                plt.scatter(self.nodes[nodeID].position[0], 
                            self.nodes[nodeID].position[1], 
                            color = 'red', 
                            s = 5)
        plt.show()
