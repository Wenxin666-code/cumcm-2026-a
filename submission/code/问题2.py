"""问题2：变物性全过程温湿场与result2.xlsx。

输入：题目附件、模型配置。
输出：固定域完整轨迹、细网格与终点资料、逐秒结果表。
同一计算目录中的轨迹可由问题3直接复用。
"""
# 绘图入口：plot模式自动计算前置模型和研究情景，再整理数据并绘制原图。
from 公共依赖 import main


def jobs(c):
    return [c.baseline_job('p23'), c.endpoint_job('p23'), *c.output_jobs(2)]


if __name__ == '__main__':
    raise SystemExit(main(2, jobs, __doc__))
