# A题论文支撑材料

本包包括四问主计算、研究扩展、数学证明与独立数值验证、题目附件、结果工作簿和科研绘图实现。程序用途与数据关系见《支撑程序输入输出说明.md》，逐文件用途见`code/data/complete_program_index.json`，文件校验见`MANIFEST.sha256.json`。

## 目录与内容

| 路径 | 内容 |
| --- | --- |
| `code/问题1.py`至`问题4.py` | 四问任务入口、结果导出和相关绘图 |
| `code/公共依赖.py` | 共享模型、求解、来源校验、工作簿处理及七张科研图 |
| `code/深度研究/` | 二维有限体积/有限元、内部收缩运动、潜热、首次事件与连续误差研究 |
| `code/config/`、`code/data/` | 模型参数、精度判据、程序索引及潜热参考数据 |
| `inputs/附件/` | 环境、半径观测与四份结果模板 |
| `results/` | 论文对应的`result1.xlsx`至`result4.xlsx` |
| `code/*验证.py`、`code/验证公共接口.py` | 七类验证程序与共享读写、路径、来源身份接口 |
| `code/data/验证数据/` | 舍入证书、晚期精确初值与根区间、安全时间上界、GLL数组及二维验证证据 |
| `AI工具使用详情.pdf` | AI工具使用详情 |

## 运行环境

主计算环境为Windows 64位、Python 3.12，第三方库版本列于`requirements.txt`。绘图采用SimHei字体。使用选定的Python解释器，在支撑材料目录中执行：

```powershell
python -m pip install -r requirements.txt
python -B code/公共依赖.py --environment --profile core --probe-imports
python -B code/问题1.py --mode check
python -B code/问题2.py --mode check
python -B code/问题3.py --mode check
python -B code/问题4.py --mode check
```

`check`核对源码身份及已交付结果文件的哈希。模型求解、数学证书复核与输出文件校验各有对应入口，具体范围见下文。

## 四问计算

以下命令将计算记录保存在支撑包外的`计算工作`目录中。首次运行可使用一个尚未建立的目录。问题2与问题3共享固定域计算，四问也可使用同一来源绑定的工作目录。

```powershell
python -B code/问题1.py --mode run --work ../计算工作
python -B code/问题2.py --mode run --work ../计算工作
python -B code/问题3.py --mode run --work ../计算工作
python -B code/问题4.py --mode run --work ../计算工作
```

默认`plan`模式展示任务及命令。`run`依次执行计算、数值对照、结果导出和读回检查。主计算采用论文配置：问题1为N=10240，问题2/3最终细网格为N=2560，问题4为N=5120。

程序通过源码、输入、配置、运行环境与输出哈希识别已完成任务。相同身份的任务可复用；输入或源码变化时采用新的工作目录。各任务的完成记录、轨迹和中间数据保存在工作目录中。结果使用未舍入数值场判定阈值，工作簿按四位小数显示。

## 科研图片

七张科研图按“附件与模型配置→主模型及研究情景→绘图数据→PDF和PNG”的流程生成：

```powershell
python -B code/公共依赖.py --plot-paper --work ../计算工作 --figure-profile production
```

各题也提供`--mode plot`入口。问题1对应环境输入、预热分布和气相边界诊断；问题2/3对应固定域过程；问题4对应收缩过程、物性与几何对照、内部运动情景。环境输入图中的终点标记取自问题2/3和问题4的计算。

`paper_figures`保存绘图数据，`figures`保存图件。`--plot-out`可指定图件目录。`--figure-profile smoke`采用低网格进行接口检查，数据和图件分别保存在`paper_figures_smoke`与`figures_smoke`中。论文精度的图件使用production配置。

完整交付目录中的`output`提供四张结构示意图的LaTeX源码和七张科研图的调用入口；`论文LaTeX源码`包含排版所需图件及论文构建程序。

## 验证程序

验证代码按数学任务组织为七个文件。阅读时先看各文件开头的流程说明，再看相应模型类与`main`入口；同目录的`验证公共接口.py`集中处理路径、JSON/CSV/压缩日志读写、数组加载、来源身份、数据校验和已有模型接口。

| 文件 | 负责的验证 | 直接复用的实现 |
| --- | --- | --- |
| `code/问题一舍入验证.py` | 温度与部分含水率的正确舍入、谱根隔离、解析区间和数值字段对表 | 共享文件与来源记录接口；严格谱算法独立实现 |
| `code/晚期区间验证.py` | 精确晚期初值、半离散误差管、连续样条屏障、指定时刻查询和局部轨迹生产 | 主模型构造、物性及数值右端 |
| `code/时间界限验证.py` | 332 h/763 h充分时间上界、累计下屏障及条件不可达性 | `code/深度研究/连续PDE.py`的有理向外算术 |
| `code/一维GLL验证.py` | 一维三次GLL独立弱式、九组求解、FV场配对和表格逐格舍入比较 | 数据和记录接口；弱式、物性保留独立实现 |
| `code/二维验证.py` | 解析/制造解、二维物理计算、全场比较、一维基准差及生产计划 | `code/深度研究/二维模型.py`的空间模型、时间推进、场记录和多项式区间算法 |
| `code/输入与假设验证.py` | 环境平台识别、24种单因素情景、物性与几何交叉及精度比较 | 主模型、边界、物性、事件求解和移动半径采样 |
| `code/一维FV验证.py` | 一维有限体积制造解、守恒结构及固定半径退化检查 | 主模型的径向与收缩求解接口 |

下面是直接可用的证据检查命令，均在支撑材料目录执行：

```powershell
python -B code/问题一舍入验证.py --mode verify --out ../核验_问题一
python -B code/晚期区间验证.py --mode verify --out ../核验_晚期
python -B code/时间界限验证.py verify --out ../核验_时间界限
python -B code/一维GLL验证.py readback --out ../核验_GLL
python -B code/二维验证.py readback --out ../核验_二维
```

重新计算各项验证的入口如下：

```powershell
python -B code/晚期区间验证.py --mode weighted --case both --out ../晚期加权区间
python -B code/晚期区间验证.py --mode continuous-first --case both --out ../晚期连续首段
python -B code/时间界限验证.py upper --out ../安全时间上界
python -B code/时间界限验证.py lower --out ../条件不可达下屏障
python -B code/一维GLL验证.py solve --out ../GLL完整求解
python -B code/二维验证.py mms --out ../二维制造解
python -B code/二维验证.py plan --out ../二维生产计划
```

完整谱根、解析区间和连续全窗口的命令见`code/验证程序说明.md`。每个文件均提供`--help`。`--out`应指向整个支撑材料目录之外的新目录，计算轨迹和输出报告保存在该目录。

`code/data/验证数据`保存证书、精确初态、原始GLL数组和来源记录，目录层次对应这些证据内部的相对路径。`code/data/验证数据清单.json`核对实际数据；`来源源码索引.json`解释既有证据中的源码身份。每次运行另外记录当前各类验证、共享接口和实际主模型的源码身份。证明中的初始条件、数值数组及参数哈希均按各验证的约定核对。

问题一证书覆盖37,800个温度值及17,357个含水率值；晚期证书针对保存节点定义的初值问题；时间界限适用于正文声明的B0一维模型及正则性条件。GLL对照、二维制造解和保存场比较分别说明数值验证范围。

## 文件校验

`MANIFEST.sha256.json`的`files`记录以支撑包为基准的文件路径及SHA-256，覆盖清单自身以外的实际文件；`main_sources`列出五份主计算源码。`code/data/验证数据清单.json`绑定证据与数组；运行报告中的`source_identity`绑定当前验证代码、主模型和研究实现。

论文、支撑材料压缩包、论文LaTeX源码及`output`各有明确用途。参赛电子文件为论文PDF与支撑材料ZIP，具体提交方式按比赛要求执行。

验证数据中的推导与运行记录保留生成时的路径语境。当前程序路径与调用方式以`code/验证程序说明.md`及《支撑程序输入输出说明》为准。
