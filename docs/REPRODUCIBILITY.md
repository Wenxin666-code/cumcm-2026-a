# 运行与复现说明

## 1. 归档核验

在仓库根目录运行：

```bash
python -B tools/check_archive.py
```

核验涵盖`submission/MANIFEST.sha256.json`中的全部202项记录、额外文件检查、21份Python源码的语法，以及四问各自的`check`和`plan`入口。计划必须返回`PDE_started: false`。这些步骤只检查归档和接口，不启动正式求解，也不证明所有数值或理论结论正确。

## 2. 依赖环境

原支撑包指定Windows 64位、Python 3.12及固定版本依赖。请使用隔离环境安装：

```bash
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
python -m pip install -r submission/requirements.txt
cd submission
python -B code/公共依赖.py --environment --profile core --probe-imports
```

数学区间验证另需`python-flint`，绘图另需`matplotlib`、`Pillow`及SimHei字体。不要修改固定版本来绕过环境核对；如需移植，应另行记录环境差异并重新核验。本次归档仅运行标准库核验，没有重新安装整个历史计算环境。

## 3. 四问计算

下列命令均在`submission/`内执行。`../work`应为包外计算目录。

```bash
python -B code/问题1.py --mode run --work ../work
python -B code/问题2.py --mode run --work ../work
python -B code/问题3.py --mode run --work ../work
python -B code/问题4.py --mode run --work ../work
```

问题2和3共享固定域轨迹。同一输入、源码及配置身份下可以复用工作目录；变更源码或输入后使用新的目录。正式网格为问题1的N=10240、问题2/3的N=2560、问题4的N=5120。完整计算有较高时间和资源需求，CI不执行求解。

## 4. 验证与绘图

已有证据的核对入口示例：

```bash
python -B code/问题一舍入验证.py --mode verify --out ../checks/q1
python -B code/晚期区间验证.py --mode verify --out ../checks/late
python -B code/时间界限验证.py verify --out ../checks/time-bound
python -B code/一维GLL验证.py readback --out ../checks/gll
python -B code/二维验证.py readback --out ../checks/2d
```

`--out`使用新的包外目录。证书读取、保存场核对和重新求解是不同操作，完整说明见[验证程序说明](../submission/code/验证程序说明.md)。连续问题证书中的精确初值、恒温、正则性等限定必须与论文一起阅读。

科研图生成命令：

```bash
python -B code/公共依赖.py --plot-paper --work ../work --figure-profile production
```

绘图会自动运行缺失的前置计算，不能当作只读取现有结果的轻量操作。`smoke`配置只用于低网格接口检查，不能替代论文精度结果。

## 5. 路径与结果

原包内部的相对路径保持不变。历史证据里的路径字符串仅记录生成时的上下文，不是本机目录要求。部分原始说明提及LaTeX构建目录、`output/`或完整交付目录，这些不在当前文件夹的常规文件中，本仓库不声称包含对应材料。

已归档的四份结果工作簿位于`submission/results/`。核验原件时以清单哈希为准；重算结果和新日志应存入`work/`或仓库外目录。
