"""问题4：材料坐标收缩模型、干燥终点与result4.xlsx。

输入：温湿环境、实测半径及题定物性配置。
输出：移动域温湿轨迹、干燥时刻、固定半径与真实表面含水率。
域外位置保持空值；收缩运动学及二维扩展研究程序另行保留。
"""
# 绘图入口：plot模式自动计算前置模型和研究情景，再整理数据并绘制原图。
from 公共依赖 import main


def jobs(c):
    return [c.baseline_job('q4'), c.endpoint_job('q4'), *c.output_jobs(4)]


if __name__ == '__main__':
    raise SystemExit(main(4, jobs, __doc__))
