# 全初值上屏障的复现

这是一组不启动 PDE 的标量有向算术计算，针对原 B0 α=1 有效模型。核心输出为 P23 332 h、P4 763 h 的条件性安全粗上界，均保证比较函数控制 Cmax≤0.149。解存在及规定正则性为前提；不覆盖新潜热模型，不替代原正式数值时间或晚期窄证书。

依赖 Python 3.12、NumPy（含 trapezoid）、openpyxl。无新增依赖，不需要 mpmath、gmpy2、SciPy 求解、字体或 LaTeX。所有路径通过 CLI 给定，可从任意 cwd 运行；code 三文件需同目录。

~~~~text
python /path/math_upper/code/certify_mass_upper.py --inputs /path/附件 --contract /path/model.json --model-code /path/code/solvers --out /writable/results
python /path/math_upper/code/verify_upper_outputs.py --results /writable/results --contract /path/model.json --out /writable/serialized_schedule_audit.json
~~~~

默认标量积分包围相对间隙为 0.0001。它控制选定屏障速率律的积分包围，绝不等同于 PDE 误差容限。所有有限形状候选的比较时间也由严格上和得到；不是对 a 的全局最优认证。

已实际完成：两套形状搜索及标量分箱至 65536/32768 箱；精确有理 exp/log 余项包围；98,304 行 CSV 全量回读；每一行的正速率、两项充分界、时长乘速率不低于对数上界、累计和与取整安全小时逐项核验。另已实际执行两个 CLI 的 --help，并做源码语法检查。所有数值与源码/附件/config SHA 在摘要中；serialized_schedule_audit.json 的 passed=true 只对应这些导出与算术核查，未声明独立完整定理复核或 PDE 存在性证明。

P23_certified_bins.csv 约34.4 MB、P4_certified_bins.csv 约17.2 MB，均保留本次真实逐箱记录。若整合至有体积上限的支撑包，可纳入这三份源码、数学报告、小摘要与 CSV SHA 索引，完整 CSV 由本入口再生；不得宣称未打入包内的大 CSV 已随包交付，也不得把旧 PDE 数组混成这次标量计算数据。

报告的字面时间由真实已完成摘要给出；整数小时向上取整。箱内采用“下降至下端即保持”的连续分段屏障，再按有向时长衔接，避免向上舍入时长导致幅度跑出已证明的箱区间。最后 U=0.098 保持不变，故整数安全时刻后的结论仍成立。
