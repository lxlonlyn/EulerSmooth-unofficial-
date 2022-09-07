r"""
    描述：
        提供向量计算的一些基本函数。
"""
from math import atan, cos, pi, sqrt
import numpy as np
import logging


def isPointOnSegment(point1, point2, pointQ) -> bool:
    """
        描述：
            判断点Q是否在point1,point2组成的线段上。

        参数：
            - point1((float,float)): 第一个线段端点坐标
            - point2((float,float)): 第二个线段端点坐标
            - pointQ((float,float)): 查询坐标
    """
    x1, y1 = point1
    x2, y2 = point2
    xQ, yQ = pointQ
    maxX = max(x1, x2)
    maxY = max(y1, y2)
    minX = min(x1, x2)
    minY = min(y1, y2)
    if (abs(((xQ - x1) * (y2 - y1) - (x2 - x1) *(yQ - y1))) < 0.00001) and (xQ >= minX and xQ <= maxX) and (yQ >=minY and yQ <= maxY):
        return True
    else:
        return False

def distance(p1: np.ndarray, p2: np.ndarray = None) -> float:
    r"""
        描述：
            计算两个端点间距离。
        
        参数：
            - p1(ndarray): 第一个端点位置
            - p2(ndarray): 第二个端点位置

        返回值：
            distance(float): 两点间距离
    """
    if p2 is None:
        p2 = np.zeros(2)
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculateAngle(p: np.ndarray) -> float:
    r"""
        描述：
            返回向量的角度（弧度制），范围为[0,2*PI)

        参数：
            - p(ndarray): 向量

        返回值：
            angle: 角度（弧度制）
    """
    x, y = p
    if x > 0 and y >= 0:
        return atan(y / x)
    elif x > 0 and y < 0:
        return atan(y / x) + 2 * pi
    elif x < 0:
        return atan(y / x) + pi
    elif abs(x) < 1e-5 and y > 0:
        return pi / 2
    elif abs(x) < 1e-5 and y < 0:
        return pi * 3 / 2
    else:
        logging.warning("警告：无法转换角度")
        return 0.0

def calculateAngleDiff(a: float, b: float) -> float:
    r"""
        描述：
            计算两个角度的差值的绝对值，范围在[0, PI]
        
        参数：
            - a(float): 第一个角度
            - b(float): 第二个角度

        返回值：
            diff(float): 两个角度的差值的绝对值，范围在[0, PI]
    """
    if a < b:
        t = a
        a = b
        b = t
    diff = a - b
    while diff > pi:
        diff -= 2 * pi
    if diff < 0:
        diff = -diff
    return diff

def calculateMaxMovement(collisionAngle: float, moveAngle: float, collisionDistance: float) -> float:
    r"""
        描述：
            计算在moveAngle角度上移动多长距离能使其在collisionAngle上投影距离与collisionDistance相同。
            若两者夹角不小于 PI/2，则返回 INF

        参数：
            - collisionAngle(float): 被投影角度
            - moveAngle(float): 待计算长度的角度
            - collisionDistance: 被投影的具体长度 
        
        返回值：
            moveDistance(float): 需要移动的距离
    """
    angleDiff = calculateAngleDiff(collisionAngle, moveAngle)
    if angleDiff < pi / 2:
        return collisionDistance / cos(angleDiff)
    else:
        return float('inf')

def isPointInTriangle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, o: np.ndarray) -> bool:
    r"""
        描述：
            返回点o是否在p1,p2,p3围成的三角形范围内。

        参数：
            - p1(ndarray): 第一个三角形顶点的坐标
            - p2(ndarray): 第二个三角形顶点的坐标
            - p3(ndarray): 第三个三角形顶点的坐标
            - o(ndarray): 待判断的点
        
        返回值：
            isPointInTriangle(bool): 点o是否在p1,p2,p3围成的三角形范围内
    """
    if np.cross(p2 - p1, p3 - p1) < 0:
        return isPointInTriangle(p1, p3, p2, o)
    elif np.cross(p2 - p1, o - p1) > 0 and np.cross(p3 - p2, o - p2) > 0 and np.cross(p1 - p3, o - p3) > 0:
        return True
    else:
        return False
