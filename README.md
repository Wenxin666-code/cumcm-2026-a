# 2026 CUMCM A题：药材热湿传递与收缩干燥建模

2026年“高教社杯”全国大学生数学建模竞赛A题团队项目归档。

**论文题目**：基于干基守恒的药材热湿传递与收缩干燥建模

**获奖记录**：上海赛区本科组三等奖

**整理维护**：[张艺严](https://github.com/Wenxin666-code)

[阅读论文](paper/report.pdf) · [支撑包说明](submission/README.md) · [输入输出说明](submission/支撑程序输入输出说明.md) · [运行与复现](docs/REPRODUCIBILITY.md) · [获奖公示](https://dxs.moe.gov.cn/zx/a/hd_sxjm_smkx/261111/2060187.shtml)

## 项目内容

围绕圆柱形药材的热风烘干过程，建立温度与干基含水率的传递模型，分析全域干燥终点，以及半径收缩对过程和时长的影响。

| 任务 | 内容 | 程序入口 | 已归档结果 |
| --- | --- | --- | --- |
| 问题1 | 固定物性下前1800秒的预热温湿分布 | [问题1.py](submission/code/问题1.py) | [result1.xlsx](submission/results/result1.xlsx) |
| 问题2 | 固定几何域下的变物性热湿传递全过程 | [问题2.py](submission/code/问题2.py) | [result2.xlsx](submission/results/result2.xlsx) |
| 问题3 | 基于全域含水率阈值确定干燥终点 | [问题3.py](submission/code/问题3.py) | [result3.xlsx](submission/results/result3.xlsx) |
| 问题4 | 耦合半径收缩后重新计算温湿场与干燥终点 | [问题4.py](submission/code/问题4.py) | [result4.xlsx](submission/results/result4.xlsx) |

计算使用有限体积离散与数值时间积分。扩展程序包含二维模型、收缩运动学、潜热及事件研究。验证材料包括制造解、独立格式比较、区间证书和条件性时间界限。各项证据的适用模型、假设和初值条件以论文与验证说明为准。

## 目录

```text
.
├── README.md
├── paper/
│   └── report.pdf                    # 论文原件（含附录）
├── submission/                       # 原始支撑包，内部路径及文件内容保留
│   ├── README.md
│   ├── requirements.txt
│   ├── MANIFEST.sha256.json
│   ├── AI工具使用详情.pdf
│   ├── code/                         # 四问入口、共享模型及验证程序
│   │   ├── config/                   # 模型与数值判据
│   │   ├── data/                     # 参考数据、证书及已有验证证据
│   │   └── 深度研究/                  # 扩展模型与研究入口
│   ├── inputs/附件/                  # 赛题输入和结果模板
│   └── results/                      # 四份计算结果工作簿
├── docs/
│   ├── REPRODUCIBILITY.md             # 环境、核验与计算说明
│   └── ARCHIVE.md                     # 整理范围与材料来源
├── tools/check_archive.py             # 标准库归档核验，不启动求解
└── .github/workflows/archive-check.yml
```

`submission/`保留原支撑包的目录结构与SHA-256清单。请将新计算输出放在该目录之外，避免覆盖已归档结果。`result2.xlsx`约26.2 MiB，网页预览可能不可用，可下载查看。

## 快速核验

先克隆仓库：

```bash
git clone https://github.com/Wenxin666-code/cumcm-2026-a.git
cd cumcm-2026-a
```

Python 3.11及以上即可执行归档核验，无需安装科学计算依赖：

```bash
python -B tools/check_archive.py
```

该命令检查原包文件哈希、Python语法、四问结果文件及计划入口。它不重新求解模型，不能代替数值复现实验。

## 计算环境与运行

原材料记录的计算环境为Windows 64位、Python 3.12，依赖版本保留在`submission/requirements.txt`。绘图使用SimHei字体，部分资源检查使用Windows接口。

```bash
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
python -m pip install -r submission/requirements.txt
cd submission
python -B code/公共依赖.py --environment --profile core --probe-imports
python -B code/问题1.py --mode plan --work ../work
```

确认环境后，将`--mode plan`改为`--mode run`启动计算；问题2至4使用对应入口。正式网格计算及部分数学验证耗时较长，完整命令和边界见[复现说明](docs/REPRODUCIBILITY.md)。

## 材料与使用说明

这是团队参赛项目的整理归档，论文、结果和AI工具使用说明均按本地材料保存，归档核验不代表已重新计算或独立确认全部科研结论。赛题附件来自赛事提供的材料，潜热参考表的来源记录保存在其[清单](submission/code/data/latent/manifest.json)中。

仓库公开供阅读与交流，暂未指定统一开源许可证；公开可访问不代表对论文、团队代码和第三方材料统一授予再分发或其他使用许可。
