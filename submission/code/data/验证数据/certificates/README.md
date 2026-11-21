# Q1 数学证书补充材料

本目录服务于已交付四表中的 Q1。温度证书覆盖 37,800 个题定值；水分证书覆盖 17,357 个题定值，均舍入为 2.5500。其余 20,443 个 Q1 水分值，以及 Q2—Q4，并未由本目录获得正确舍入证明。覆盖数量不是工作时间比例，也不代表真实药材的实验精度。

`archive/` 保留完整必要源码、参数、原始附件、谱尾说明、存在性论证、独立审查和根任务验收。候选记录里的历史 `false` 原样保留；后来的接受事实由各自 `root_acceptance_01/acceptance.json` 说明。`archive_inventory.json` 分别记录原件与匿名副本 SHA。路径字符串采用 RESEARCH、WORKSPACE、SCRATCH、USER_HOME 角色名，JSON 重新排版；科学量与布尔值未改变。归档源码是历史材料，不能直接把其中角色路径当作本机有效路径执行。

## 默认读回

在任何普通 Python 3.10+ 环境中运行：

```
python certificates/reproduce_certificates.py
```

该命令仅核对随包文件 SHA、全部 75,600 个紧凑区间及舍入结果、256/768 组 J0/Robin 根表中保存的符号与身份关系、后续验收的覆盖范围。不需要 NumPy、SciPy 或 Arb，不启动 PDE，不启动已取消的全径向链，也不重新证明原始数学定理。可加 `--package certificates --out new_check_directory` 保存本次报告；输出目录必须新建且位于证书目录之外。

可显式传入 `--inputs path/to/附件1.xlsx --selected-q1 path/to/solution.npz`，额外核对冻结原件 SHA。没有这些参数时，仍完整读回包内证据，但不声称重新读取外部未舍入 NPZ。

## 紧凑区间的含义

`evidence/*_outward.csv.xz` 保存每个查询的时间、半径列、十进制位数、整数上下界、已证明 bin、选定数值 bin 与 binary64 十六进制值。真实上下界为整数除以 `10**decimal_places`。转换器用 Fraction 对原精确有理区间向外取整，从 12 位起自适应增加，逐项验证包含关系与原四位舍入 bin 不变。`conversion_receipt.json` 是实际检查全部 75,600 项后留下的转换收据。

这是更宽、仍有效的证书区间，不是原诊断日志的无损压缩。原精确端点与诊断日志保留在研究存档；原日志 SHA 和完整精确投影 SHA 随收据保存。`.xz` 对当前 CSV 自身是无损压缩。转换过程没有改变题定结果、参数或模型。

拥有原精确日志时，可在新目录重新逐项检查包含关系并生成字节相同的紧凑产品：

```
python certificates/reproduce_certificates.py --mode rebuild-compact --temperature-log temperature_intervals.jsonl.gz --moisture-log all_query_bounds.jsonl.gz --out fresh_compact
```

此入口先检查两份日志的原 SHA，拒绝错源；再检查全部原区间、包含关系、全部已证 bin、精确投影 SHA 及紧凑产品 SHA。

## 显式解析重建入口

完整谱根/排零日志和逐值精确区间可由 `portable_reconstruction.py` 重建。它显式装载 SHA 已核对的归档函数体，跳过历史机器路径初始化，并记录新入口及实际运行库身份；不改写旧执行日志，也不声称新入口就是旧执行源码。调用者需自行显式选择以下模式，默认不会运行。

```
python certificates/reproduce_certificates.py --mode runtime-check --arb-runtime path/to/pythonlibs --out fresh_runtime_check
python certificates/reproduce_certificates.py --mode reconstruct-roots --arb-runtime path/to/pythonlibs --out fresh_roots
python certificates/reproduce_certificates.py --mode regenerate-exact --arb-runtime path/to/pythonlibs --inputs path/to/附件1.xlsx --selected-q1 path/to/solution.npz --out fresh_exact
```

若正常 Python 环境已安装依赖，可省略 `--arb-runtime`。需要 `python-flint==0.9.0`，以及 NumPy、SciPy；历史完整版本与文件 SHA 见原 `source_bindings.json`，本次实际装载版本见验证报告。原 Windows wheel 为 `python_flint-0.9.0-cp310-abi3-win_amd64.whl`，SHA256 `8f1059536b7393e48b1444894c3b54ce1b961e4aa4a356e19217b26690f20db0`。运行库本身不随包复制。来源可由 Python Package Index 的 python-flint 0.9.0 发布页取得；无需把其它平台文件伪装为这个 Windows wheel。

`runtime-check` 仅装载归档数学函数并进行精确初等值小检查。`reconstruct-roots` 显式重建全部根、每个排零小区间及符号日志。`regenerate-exact` 进一步重建所有 T 区间以及 erfc、I0 和线性比较合并后的 C 区间，并比较原精确端点/坐标/bin/选定十六进制值的完整投影 SHA；不以 gzip 时间头的不同冒充数值不同，也不把区间重放当作新的正式接受。若重建不一致，保留新输出并报错，不覆盖原材料。

本次交付包装实际执行范围以 `packaging_validation.json` 为准：完整读回、原日志转换重建、入口和实际源 SHA 正负例及运行库装载；未在本次包装中再次执行完整解析认证或 PDE。完整重建入口本轮只验证参数/装载接口，不提前写成全程重建通过。

## 未随提交 ZIP 放入的大日志

完整排零 Arb 值日志、原精确逐值投影以及 C02/C03 的大诊断日志留研究存档。包内根表为原数值记录的无损编码，保留全部根端点、符号与导数区间；不包含全部排零叶子的 Arb 值。默认读回因此不重新检查全部排零覆盖。完整源程序及上面的显式重建入口保留，不能将这些外置日志写成已随 ZIP 附带。

全径向扩展依用户指令停止；仅保留方法源码、初始两步及小检查记录和取消摘要。其目标 1 秒未达到，没有新增题定值证书。任何上述模式都不会恢复这条链。历史源码用于说明已开展方法，后续研究需另行授权。
