"""五项研究的统一入口，默认显示模块列表。

pde：原生稠密轨迹、连续质量误差管、全空间局部屏障。
latent：潜热耦合20任务矩阵。
events：首次事件覆盖与近临界24任务矩阵。
2d：二维FV/FE、轴向运动与热量审计。
kinematics：非仿射收缩、局部时间推进与收敛矩阵。

本目录含8份代码和两份附件；配置与潜热数据统一位于code/config和code/data。公共接口.py默认导入上一级目录中的
公共依赖.py，直接适配其_initialize_science和_sections。
附件按本目录定位，配置、潜热表及事件协议按共享code目录定位。
也可通过--code-dir、--research-root或对应环境变量覆盖位置。
二维、运动学及连续PDE的输入和配置默认使用随包资源。

运行需要NumPy、SciPy、openpyxl、XlsxWriter；局部连续屏障另需
python-flint。环境依赖见支撑包根目录requirements文件。
新计算使用新的输出目录；续算和验证需要本次运行生成的轨迹、
检查点及其他记录，不继承旧报告的验收结论。
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path


MODULES = {
    "pde": ("连续PDE", "原生轨迹、连续质量误差管、全空间局部屏障"),
    "latent": ("潜热矩阵", "潜热耦合 20 任务矩阵"),
    "events": ("事件覆盖", "首次事件覆盖、近临界 24 任务矩阵"),
    "2d": ("二维模型", "二维 FV/FE、轴向运动与热量审计"),
    "kinematics": ("收缩运动学", "非仿射收缩、局部时间推进与收敛矩阵"),
}


def main(argv=None):
    """仅配置共享资源并转交模块 main，不复制各模块的参数或实现。"""
    descriptions = "\n".join(
        f"  {key:<12} {name}.py：{description}"
        for key, (name, description) in MODULES.items()
    )
    parser = argparse.ArgumentParser(
        description="五项保留研究功能的普通 Python 入口。\n\n" + descriptions,
        epilog="查看模块参数：python 研究入口.py 模块名 --help\n"
        "资源与环境说明见本文件顶部文档字符串。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--code-dir", type=Path, help="包含原公共依赖.py 的 code 目录")
    parser.add_argument("--research-root", type=Path, help="深度研究资源目录")
    parser.add_argument("module", nargs="?", choices=MODULES, help="要运行的研究模块")
    parser.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="原样交给模块的参数"
    )
    args = parser.parse_args(argv)
    if args.module is None:
        parser.print_help()
        return None

    from 公共接口 import configure

    configure(args.code_dir, args.research_root)
    module = import_module(MODULES[args.module][0])
    return module.main(args.arguments)


if __name__ == "__main__":
    main()
