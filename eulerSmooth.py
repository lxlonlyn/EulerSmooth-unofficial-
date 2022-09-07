r"""
    描述：
        对图进行EulerSmooth操作。
"""
from math import pi
from random import random
import numpy as np
from utils import *
from graph import *


class EulerSmooth:
    r"""
        描述：
            用于对图进行EulerSmooth操作。
        
        属性：
            - graph(ExtendedGraph): 图
            - forces(dict[str, np.ndarray]): 结点所受力
            - constraints(dict[str, float]): 结点的限制
            - lastAngle(dict[str, float]): （MovementAcceleration）上一次迭代计算角度 
            - lastMovement(dict[str, float]): （MovementAcceleration）上一次迭代计算限制(ci)
    """
    graph: ExtendedGraph
    forces: Dict[str, np.ndarray]
    constraints: Dict[str, float]
    lastAngle: Dict[str, float]
    lastMovement: Dict[str, float]

    def __init__(self, g: ExtendedGraph) -> None:
        r"""
            描述：
                构造EulerSmooth类并初始化。

            参数：
                - g(ExtendedGraph): 图
        """
        self.graph = g
        self.forces = {}
        self.constraints = {}
        self.lastAngle = {}
        self.lastMovement = {}
        for nodeID in self.graph.nodes:
            self.forces[nodeID] = np.zeros(2, dtype=float)
            self.constraints[nodeID] = float('inf')
    
    def iterater(self, d: float, showChange: bool = False) -> None:
        r"""
            描述：
                执行一次迭代操作。

            参数：
                - d(float): 距离参数
                - showChange(bool): 是否在图上标注位置变化
        """
        # 初始化
        self.forces.clear()
        self.constraints.clear()
        for nodeID in self.graph.nodes:
            self.forces[nodeID] = np.zeros(2, dtype=float)
            self.constraints[nodeID] = float('inf')
        # 计算力
        self.computeForces(d, False, False)
        # 计算限制
        self.computeConstraints(d, False, True)
        # 移动结点，更新图
        self.moveNodes(showChange)
        # 后续处理
        self.processFlexibleEdges(1.45 * d, 1.5 * d)
        # self.processFlexibleEdges(float('inf'), 1.5 * d)
        print(len(self.graph.nodes), len(self.graph.edges))
        # 展示新图
        self.graph.show()
    
    def moveNodes(self, showChange: bool = False) -> None:
        r"""
            描述：
                根据计算结果移动结点。

            参数：
                - showChange(bool): 是否要在图上标注变化
        """
        for nodeID in self.forces:
            force = self.forces[nodeID]
            constraint = self.constraints[nodeID]
            if distance(force) > constraint:
                force = force * constraint / distance(force)
            if showChange:
                plt.quiver(
                    self.graph.nodes[nodeID].position[0], 
                    self.graph.nodes[nodeID].position[1], 
                    force[0], force[1])
            self.graph.nodes[nodeID].position += force

    #####################
    # 以下部分为力的计算 #
    #####################

    def computeForces(self, idealDistance: float, 
                            isElementMoveable: bool,
                            separateBoundaries: bool) -> None:
        r"""
            描述：
                计算每个结点所受的力。

            参数：
                - idealDistance(float): 理想距离
                - isElementMoveable(bool): 集合内元素是否可移动
                - separateBoundaries(bool): 边界是否可拆分
        """
        self.computeForceCurveSmoothing()
        self.computeForceEdgeNodeRepulsion(idealDistance, True)
        self.computeForceEdgeContraction(0.7 * idealDistance)
        if isElementMoveable:
            self.computeForceNodeNodeRepulsion(idealDistance)
        if separateBoundaries:
            self.computeForceEdgeNodeRepulsion(idealDistance / 15, False)

    def computeForceEdgeContraction(self, d: float):
        r"""
            描述：
                计算EdgeContraction力。

            参数：
                - d(float): 理想线段长度
        """
        for edgeID in self.graph.edges:
            edge = self.graph.edges[edgeID]
            uID, vID = edge.extremities
            uNode = self.graph.nodes[uID]
            vNode = self.graph.nodes[vID]
            pu = uNode.position
            pv = vNode.position
            force = distance(pu, pv) / d * (pv - pu)
            self.forces[uID] = self.forces[uID] + force
            self.forces[vID] = self.forces[vID] - force

    def computeForceNodeNodeRepulsion(self, d: float):
        r"""
            描述：
                计算NodeNodeRepulsion力。
            
            参数：
                - d(float): 理想结点间长度
        """
        # 筛选出集合内结点
        nodesID = []
        for nodeID in self.graph.nodes:
            if self.graph.nodes[nodeID].degree == 0:
                nodesID.append(nodeID)
        # 计算力
        for i in range(len(nodesID)):
            uID = nodesID[i]
            pu = self.graph.nodes[uID].position
            for j in range(i + 1, len(nodesID)):
                vID = nodesID[j]
                pv = self.graph.nodes[vID].position
                force = ((d / distance(pu, pv)) ** 2) * (pu - pv)
                self.forces[uID] = self.forces[uID] + force
                self.forces[vID] = self.forces[vID] - force

    def computeForceEdgeNodeRepulsion(self, d: float, isSetElement: bool = True):
        r"""
            描述：
                计算EdgeNodeRepulsion力。

            参数：
                - d(float): 理想结点与边距离
                - isSetElement(bool): 参与计算的结点是否是集合内结点，若不是则为边界结点参与计算
        """
        # 确定参与计算的结点与线段的集合
        if isSetElement:
            edgesID = self.graph.edges.keys()
            nodesID = []
            for nodeID in self.graph.nodes:
                if self.graph.nodes[nodeID].degree == 0:
                    nodesID.append(nodeID)
        else:
            edgesID = self.graph.edges.keys()
            nodesID = []
            for nodeID in self.graph.nodes:
                if self.graph.nodes[nodeID].degree != 0:
                    nodesID.append(nodeID)
        # 计算过程
        for edgeID in edgesID:
            edge = self.graph.edges[edgeID]
            vID, wID = edge.extremities
            pv = self.graph.nodes[vID].position
            pw = self.graph.nodes[wID].position
            for uID in nodesID:
                if uID == vID or uID == wID:
                    continue
                pu = self.graph.nodes[uID].position
                # 计算投影点
                unitVector = (pw - pv) / distance(pw, pv)
                pp = np.dot(unitVector, pu - pv) * unitVector + pv
                # 如果太近则跳过
                if (pu == pp).all():
                    continue
                forceu = np.array((0, 0), dtype=float)
                forcev = np.array((0, 0), dtype=float)
                forcew = np.array((0, 0), dtype=float)
                if isPointOnSegment(pv, pw, pp):
                    # 投影在线段上
                    forceu = ((d / distance(pu, pp)) ** 2) * (pu - pp) 
                    forcev = -forceu * distance(pp, pw) / distance(pv, pw)
                    forcew = -forceu * distance(pp, pv) / distance(pv, pw)
                else:
                    # 投影在线段外
                    if distance(pu, pv) > distance(pu, pw):
                        pn = pv
                    else:
                        pn = pw
                    forceu = ((d / distance(pu, pn)) ** 2) * (pu - pn)
                    if (pn == pv).all():
                        forcev = -forceu
                    else:
                        forcew = -forceu
                self.forces[uID] = self.forces[uID] + forceu
                self.forces[vID] = self.forces[vID] + forcev
                self.forces[wID] = self.forces[wID] + forcew

    def computeForceCurveSmoothing(self):
        r"""
            描述：
                计算CurveSmoothing力。
            
            参数：无
        """
        for subgraph in self.graph.subgraphs:
            if len(subgraph.nodesID) < 3:
                return
            lastEdge = self.graph.edges[subgraph.edgesID[-1]]
            for currentEdgeID in subgraph.edgesID:
                currentEdge = self.graph.edges[currentEdgeID]
                uID = lastEdge.getCommonExtremity(currentEdge)
                tID = lastEdge.getAnotherExtremity(uID)
                vID = currentEdge.getAnotherExtremity(uID)
                pu = self.graph.nodes[uID].position
                pt = self.graph.nodes[tID].position
                pv = self.graph.nodes[vID].position
                force = 2 / 3 * ((pt + pv) / 2 - pu)
                self.forces[uID] = self.forces[uID] + force
                lastEdge = currentEdge

    ######################
    # 以下部分为限制的计算 #
    ######################

    def computeConstraints(self, maxDistance: float, 
                                 isElementMoveable: bool,
                                 isBoundaryIndependent: bool) -> None:
        r"""
            描述：
                计算每个结点的限制。

            参数：
                - idealDistance(float): 最大限制距离
                - isElementMoveable(bool): 集合内元素是否可移动
                - isBoundaryIndependent(bool): 是否允许
        """
        self.computeConstraintDecreasingMaxMovement(maxDistance)
        self.computeConstraintMovementAcceleration(maxDistance)
        self.computeConstraintSurroundingEdges(isBoundaryIndependent)
        if not isElementMoveable:
            self.computeConstraintPinnedNodes()

    def computeConstraintDecreasingMaxMovement(self, d: float) -> None:
        r"""
            描述：
                计算DecreasingMaxMovement限制。

            参数：
                - d(float): 最大距离限制
        """
        for nodeID in self.graph.nodes:
            self.constraints[nodeID] = min(self.constraints[nodeID], d)

    def computeConstraintMovementAcceleration(self, d: float) -> None:
        r"""
            描述：
                计算MovementAcceleration限制。

            参数：
                - d(float): 初始最大距离限制
        """
        for nodeID in self.graph.nodes:
            force = self.forces[nodeID]
            if (force == np.zeros(2, dtype=float)).all():
                if nodeID in self.lastAngle:
                    self.lastAngle.pop(nodeID)
                self.lastMovement[nodeID] = 0.0
                continue
            currentAngle = calculateAngle(force)
            if nodeID not in self.lastAngle:
                currentMovement = d
            else:
                lastAngle = self.lastAngle[nodeID]
                lastMovement = self.lastMovement[nodeID]
                diffAngle = calculateAngleDiff(currentAngle, lastAngle)
                if diffAngle < pi / 3:
                    currentMovement = lastMovement * (1 + 2 * (1 - diffAngle / (pi / 3)))
                elif diffAngle < pi / 2:
                    currentMovement = lastMovement
                else:
                    currentMovement = lastMovement / (1 + 4 * (diffAngle / (pi / 2) - 1))
            self.lastMovement[nodeID] = currentMovement
            self.lastAngle[nodeID] = currentAngle
        for nodeID in self.lastMovement:
            if nodeID not in self.constraints:
                continue
            self.constraints[nodeID] = min(self.constraints[nodeID], self.lastMovement[nodeID])
        
    def computeConstraintPinnedNodes(self) -> None:
        r"""
            描述：
                计算PinnedNodes限制。（只限制集合内元素）

            参数：无
        """
        for nodeID in self.graph.nodes:
            node = self.graph.nodes[nodeID]
            if node.degree == 0:
                self.constraints[nodeID] = 0.0

    def computeConstraintSurroundingEdges(self, isBoundaryIndependent: bool) -> None:
        r"""
            描述：
                计算SurroundingEdges限制。

            参数：
                - isBoundaryIndepentdent(bool): 是否边界独立（边界结点可以跨边界）
        """
        if isBoundaryIndependent:
            # 只选取集合内结点
            nodesID = []
            for nodeID in self.graph.nodes:
                node = self.graph.nodes[nodeID]
                if node.degree == 0:
                    nodesID.append(nodeID)
        else:
            # 选取所有结点
            nodesID = list(self.graph.nodes.keys())
        edgesID = list(self.graph.edges.keys())
        for edgeID in edgesID:
            edge = self.graph.edges[edgeID]
            vID, wID = edge.extremities
            pv = self.graph.nodes[vID].position
            pw = self.graph.nodes[wID].position
            for uID in nodesID:
                # 结点是线段端点，跳过
                if uID == vID or uID == wID:
                    continue
                pu = self.graph.nodes[uID].position
                # 计算投影点
                unitVector = (pw - pv) / distance(pw, pv)
                pp = np.dot(unitVector, pu - pv) * unitVector + pv
                if isPointOnSegment(pv, pw, pp):
                    # 投影在线段上
                    collisionAngle = calculateAngle(pp - pu)
                    collisionDistance = distance(pp, pu) / 2 - 1 # 点和线段最多移动 maxDis / 2
                    if collisionDistance < 0:
                        collisionDistance = 0
                    self.constraints[uID] = min(self.constraints[uID], 
                        calculateMaxMovement(collisionAngle, calculateAngle(self.forces[uID]), collisionDistance))
                    self.constraints[vID] = min(self.constraints[vID], 
                        calculateMaxMovement(collisionAngle + pi, calculateAngle(self.forces[vID]), collisionDistance))
                    self.constraints[wID] = min(self.constraints[wID], 
                        calculateMaxMovement(collisionAngle + pi, calculateAngle(self.forces[wID]), collisionDistance))
                else:
                    # 投影在线段外
                    if distance(pu, pv) > distance(pu, pw):
                        pn = pv
                    else:
                        pn = pw
                    collisionAngle = calculateAngle(pn - pu)
                    collisionDistance = distance(pp, pu) / 2 - 1
                    if collisionDistance < 0:
                        collisionDistance = 0
                    self.constraints[uID] = min(self.constraints[uID], 
                        calculateMaxMovement(collisionAngle, calculateAngle(self.forces[uID]), collisionDistance))
                    if (pn == pv).all():
                        self.constraints[vID] = min(self.constraints[vID], 
                            calculateMaxMovement(collisionAngle + pi, calculateAngle(self.forces[vID]), collisionDistance))
                    else:
                        self.constraints[wID] = min(self.constraints[wID], 
                            calculateMaxMovement(collisionAngle + pi, calculateAngle(self.forces[wID]), collisionDistance))

    ##########################
    # 以下部分为后续处理的计算 #
    ##########################
          
    def processFlexibleEdges(self, contractDistance: float, expandDistance: float) -> None:
        r"""
            描述：
                预处理操作FlexibleEdges。

            参数：
                - contractDistance(float): 线段裁剪的阈值
                - expandDistance(float): 线段扩展的阈值
        """
        if random() > 0.5:
            contractDistance = float('inf')
        # 裁剪
        for subgraph in self.graph.subgraphs:
            if len(subgraph.edgesID) < 3:
                continue
            lastEdgeID = subgraph.edgesID[-1]
            i = 0
            while i < len(subgraph.edgesID):
                edgeID = subgraph.edgesID[i]
                lastEdge = self.graph.edges[lastEdgeID]
                curEdge = self.graph.edges[edgeID]
                uID = curEdge.getCommonExtremity(lastEdge)
                preID = lastEdge.getAnotherExtremity(uID)
                nxtID = curEdge.getAnotherExtremity(uID)
                if self.graph.nodes[uID].degree != 2:
                    # 如果是公共交点则不处理
                    lastEdgeID = edgeID
                    i += 1
                    continue
                if distance(self.graph.nodes[preID].position, self.graph.nodes[nxtID].position) < contractDistance:
                    # 距离过长不处理
                    lastEdgeID = edgeID
                    i += 1
                    continue
                existElement = False
                for nodeID in self.graph.nodes:
                    node = self.graph.nodes[nodeID]
                    if isPointInTriangle(self.graph.nodes[uID].position,
                                         self.graph.nodes[preID].position,
                                         self.graph.nodes[nxtID].position,
                                         node.position):
                        existElement = True
                        break
                if not existElement:
                    if lastEdge.extremities[0] == uID:
                        lastEdge.extremities = (nxtID, lastEdge.extremities[1])
                    elif lastEdge.extremities[1] == uID:
                        lastEdge.extremities = (lastEdge.extremities[0], nxtID)
                    else:
                        logging.warning("警告：未找到初始结点")
                    subgraph.edgesID.remove(edgeID)
                    subgraph.nodesID.remove(uID)
                    self.graph.edges.pop(edgeID)
                    self.graph.nodes.pop(uID)
                else:
                    lastEdgeID = edgeID
                    i += 1
        # 扩展
        edgeExpandList = []
        for edgeID in self.graph.edges:
            edge = self.graph.edges[edgeID]
            uID, vID = edge.extremities
            uNode = self.graph.nodes[uID]
            vNode = self.graph.nodes[vID]
            edgeLength = distance(uNode.position, vNode.position)
            if edgeLength > expandDistance:
                edgeExpandList.append(edgeID)  
        for edgeID in edgeExpandList:
            edge = self.graph.edges[edgeID]
            uID, vID = edge.extremities
            uNode = self.graph.nodes[uID]
            vNode = self.graph.nodes[vID]
            edgeLength = distance(uNode.position, vNode.position)
            subLength = edgeLength
            newEdgeNumber = 1
            # 由于每次为中点，所以最后新增线段为2的倍数
            while subLength > expandDistance:
                newEdgeNumber *= 2
                subLength /= 2
            # 加入新结点和新线段
            nodeListID = []
            nodeListID.append(uID)
            for i in range(1, newEdgeNumber):
                pos = uNode.position + (vNode.position - uNode.position) / newEdgeNumber * i
                nodeListID.append(self.graph.addNode(pos))
            nodeListID.append(vID)
            edgeListID = []
            for i in range(1, len(nodeListID)):
                edgeListID.append(self.graph.addEdge(nodeListID[i - 1], nodeListID[i]))
            # 更改子图中元素
            for subgraph in self.graph.subgraphs:
                newEdgesID = []
                newNodesID = []
                # 加入线段
                for i in range(len(subgraph.edgesID)):
                    if subgraph.edgesID[i] == edgeID:
                        if i == 0:
                            isReversed = i + 1 < len(subgraph.edgesID) \
                                         and (self.graph.edges[subgraph.edgesID[i + 1]].extremities == self.graph.edges[edgeListID[0]].extremities)
                        else:
                            isReversed = (self.graph.edges[subgraph.edgesID[i - 1]].extremities == self.graph.edges[edgeListID[-1]].extremities)
                        if not isReversed:
                            for newEdgeID in edgeListID:
                                newEdgesID.append(newEdgeID)
                        else:
                            for newEdgeID in reversed(edgeListID):
                                newEdgesID.append(newEdgeID)
                    else:
                        newEdgesID.append(subgraph.edgesID[i])
                # 加入结点
                if len(newEdgesID) > 0:
                    lastEdge = self.graph.edges[newEdgesID[-1]]
                for newEdgeID in newEdgesID:
                    curEdge = self.graph.edges[newEdgeID]
                    commonNodeID = curEdge.getCommonExtremity(lastEdge)
                    newNodesID.append(commonNodeID)
                    lastEdge = curEdge
                for nodeID in subgraph.nodesID:
                    if self.graph.nodes[nodeID].degree == 0:
                        newNodesID.append(nodeID)
                # 更新信息
                subgraph.nodesID = newNodesID
                subgraph.edgesID = newEdgesID
            # 删除线段信息
            self.graph.edges.pop(edgeID)
