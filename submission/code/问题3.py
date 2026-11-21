"""问题3：全域干燥终点与result3.xlsx。

输入：题目附件、模型配置，或问题2已完成的共同计算目录。
输出：全域最大含水率判据、严格干侧报告时刻与逐分钟结果表。
问题2已完成时复用固定域轨迹；单独运行时先完成相同计算。
"""
# 绘图入口：plot模式自动计算前置模型和研究情景，再整理数据并绘制原图。
from 公共依赖 import main


def jobs(c):
    return [c.baseline_job('p23'), c.endpoint_job('p23'), *c.output_jobs(3)]


if __name__ == '__main__':
    raise SystemExit(main(3, jobs, __doc__))
