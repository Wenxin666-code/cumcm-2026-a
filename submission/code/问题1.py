"""问题1：预热求解、紧容差轨迹、必要对照及result1.xlsx导出。

输入：附件1、附件2和模型配置。
输出：1800秒温湿轨迹、必要比较记录、result1.xlsx。
"""
# 绘图入口：plot模式自动计算前置模型和研究情景，再整理数据并绘制原图。
from 公共依赖 import main


def jobs(c):
    shared = ['--base-code', c.code, '--inputs', c.inputs, '--config', c.contract]
    return [
        c.baseline_job('q1'),
        c.job('q1_disk_equivalence', 'extensions/q1/portable_q1_disk.py',
              shared + ['--scratch', c.q1, '--mode', 'validate'],
              [c.q1 / 'equivalence_N160/equivalence.json'],
              [(c.q1 / 'equivalence_N160/equivalence.json', 'passed')],
              artifact_roots=[c.q1 / 'equivalence_N160']),
        c.job('q1_full_tight', 'extensions/q1/portable_q1_disk.py',
              shared + ['--scratch', c.q1, '--mode', 'run'],
              [c.q1 / 'full/run_status.json'],
              [(c.q1 / 'full/run_status.json', key) for key in ['solver_completed', 'sources_unchanged']],
              artifact_roots=[c.q1 / 'full']),
        c.job('q1_full_comparison', 'extensions/q1/portable_q1_compare.py',
              shared + ['--runs', c.baseline, '--full', c.q1 / 'full',
                        '--equivalence', c.q1 / 'equivalence_N160/equivalence.json',
                        '--scratch', c.runs / 'q1_full_comparison'],
              [c.runs / 'q1_full_comparison/summary.json'],
              [(c.runs / 'q1_full_comparison/summary.json', key)
               for key in ['completed', 'passed', 'comparison_completed']],
              artifact_roots=[c.runs / 'q1_full_comparison']),
        *c.output_jobs(1),
    ]


if __name__ == '__main__':
    raise SystemExit(main(1, jobs, __doc__))
