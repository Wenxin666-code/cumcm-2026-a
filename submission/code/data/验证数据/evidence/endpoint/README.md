# 终点误差控制增补的复现说明

本目录包含独立的全程细网格对照、题定输出候选、积分核验和晚期条件区间证明。原模型包始终只读。`endpoint_summary.md/json`（完成后生成）区分经验网格差、半离散条件证书、连续标量条件证书及未认证环节。文件中 `passed` 只对紧邻说明的检查对象负责，不表示原全程连续 PDE 已认证。

## 运行环境

使用 Python 3.12、NumPy、SciPy、openpyxl；实际计算使用与基础模型相同的冻结依赖。区间核使用 Python 标准库 Decimal（44 位），基本运算和正确舍入的 exp 均向外扩张；不依赖未验证的 NumPy 初等函数作为区间界。运行前设置 `PYTHONDONTWRITEBYTECODE=1`、`PYTHONUTF8=1`，`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`。入口亦设置这些变量，避免改写原基础代码目录。

`--base-code` 指向原模型包的 `code`，其中应有 `solvers/moving_model.py`、`solvers/relative_temperature.py`、`solvers/_vendor_host2_model.py` 和 `validation/absolute_temperature_check.py`。`--inputs` 指向原附件目录，`--contract` 指向原 model.json。`--runs` 是已完成原主算例目录，含 `p23/internal.npz`、`q4/internal.npz` 和各自的 endpoint.json。`--work` 保存小报告，`--scratch` 保存大数组；所有新输出目录应为空，不要指向基础模型结果目录。

## 从基础模型完整复现

```text
python code/reproduce_endpoint.py --base-code BASE/code --inputs INPUTS --contract BASE/config/model.json --runs BASE_RUNS --work NEW_REPORTS --scratch NEW_SCRATCH --stage all --case both
```

各阶段可单独调用：`local`（晚期独立重启）、`certificates`（均匀/加权半离散管及来源绑定）、`refinement`（三组全程参考设置）、`continuous`（所有时空区间连续屏障）、`candidate`（完整题定输出）、`balance`（相同全程的独立8/16阶积分）、`validation`（三组固定物理位置及全径向重放/比较）。`all` 按依赖顺序执行，不自动删除失败目录或改变网格、容差、输入节点。可通过 `--p23-refinements` 和 `--q4-refinements` 指向已完成且哈希可核的分开放置参考目录。

每个单独数值入口已经实际完整执行；总调度入口执行 `--help` 检查，其功能是按上述依赖串行调用同一源程序，不应把调度入口本身写成另一次已完成全流程。历史真实失败和恢复见 `results/attempt_ledger.json`。

## 文件与口径

- `finest_candidate/p23`：Q2逐秒21半径分块 NPZ、Q3逐分钟21半径 NPZ、六时刻/六小时论文采样、全节点分钟剖面和新旧报告时刻精确字段。
- `finest_candidate/q4`：正式21列契约为20个固定物理半径加真实表面；`result4_candidate_data.npz` 将表面单列。额外21固定半径诊断单独存储，域外为 NaN、valid_interior 为假，不挤占正式表面列。
- 两问 `endpoint.json` 记录动态报告刻度、未舍入全节点最大C、经验裕量。根不是严格不等号下的“最小时刻”，报告时刻由完整网格根差加容差根差再向干侧取刻度。
- `finest_balance/{case}/balance.json`：8/16阶独立积分及观察器无副作用核验。热积分为 Jρcp T_t；不把 JρcpT 的导数替换为热容量项。
- `validation_replay`：每问 coarse/main/tight 三次完整计算。根、接受步哈希和原样本逐位匹配；另存前3h逐秒固定物理位置、全程逐分钟完整径向场。
- `validation_metrics/{case}/convergence.json`：同容差网格差与同细网格容差差，含位置、时刻、计数、输入数组及源码哈希。只包容已声明采样集合；节点并集最大值精确对应两个分段线性重构的空间差，不是 PDE 上界。
- `proofs`：晚期 Radau 多项式、精确参数快照、均匀/加权区间证书。均匀管可能已闭合而事件未在窗口内包住；两项状态分别记录。
- `continuous/{case}/continuous_certificate.json`：通过指定晚期节点的唯一夹持C2样条初态问题、精确恒温、冻结binary64参数条件下的连续标量屏障。它不包围新细网格全程继承状态或原全程耦合解。

## 不依赖大数组的小证据包

生成命令：

```text
python code/build_small_bundle.py --proofs PROOFS --production BASE_RUNS --contract BASE/config/model.json --certificates CONTINUOUS --reports REPORTS --out SMALL_BUNDLE
```

小包包含独立区间核、两个实际晚期多项式、网格/半径/物性精确快照及 SHA-256 清单，不包含大量题定输出。仅核验晚期证书时不需要附件、基础模型代码或全程数组：

```text
python code/verify_small_bundle.py --bundle SMALL_BUNDLE --out NEW_ODE_CHECK --case both --method weighted
python code/verify_small_bundle.py --bundle SMALL_BUNDLE --out NEW_PDE_CHECK --case both --method continuous
```

第二条重新计算所有时空区间。可加 `--continuous-first-slab` 检查独立目录载入与首段数学核，但该模式明确不代表全窗口证明。程序逐一比对重新计算的证明数值，源哈希和墙钟字段不参与数值相等判定。固定节点权重从归档值直接载入，避免不同平台的 libm 将“相同数学表达式”算为不同精确输入常数。

`reviews` 归档独立审查脚本和小报告，保留实际执行源码字节与哈希。`continuous_spline_review.py` 和早期 `endpoint_certificate_review.py` 保留原研究目录相对布局，不能直接从小包新位置当作独立CLI运行；这两份是来源证据。带 `--code`、`--root` 等参数的审查入口需按各自 `--help` 指向所声明目录布局。小包自包含的证明复核使用上述 `verify_small_bundle.py`，不依赖历史审查的原路径。整证书一致性审查验证代数接口、增长上包、根符号和来源绑定，不替代另行完成的全空间残差核验。

## 尚未包含的严格证明

没有从零时刻至晚期检查点的连续误差管，也没有非恒温条件下温度梯度误差及非合作耦合稳定性的包围；输入转二进制、尾环境均值计算和物理模型误差同样未纳入。C2连续屏障针对重新指定的正则初值问题，不能跨越原问题初边值不相容层而自动沿用。后验对照、独立有限元、制造解与守恒支持工程数值可信度，不能改称连续全程认证。
