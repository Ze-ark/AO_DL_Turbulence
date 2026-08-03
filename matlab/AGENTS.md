# MATLAB目录规则

本文件适用于 `matlab/` 及其子目录，并补充仓库根 `AGENTS.md`。

## 物理实现

- MATLAB是物理参考实现。传播、相位屏、Fried参数、SLM响应和焦面指标的修改必须先保证物理定义清楚，再考虑运行速度。
- 相位屏在空间域施加，自由空间传播在频域完成；不得再次跳过第一块相位屏。
- 无湍流参考与湍流场必须传播到同一目标平面；完整路径 `r0` 与单段相位屏参数不得混用。
- 相位单位默认弧度；长度、时间、风速、频率和光功率必须在变量名、注释或元数据中注明单位。
- 相位误差只在有效瞳面内计算并去除piston。焦面性能使用独立的Strehl、桶内功率或环围能量评价。
- 使用随机数时接受显式种子，并恢复调用前的MATLAB随机状态，避免污染其他实验。

## 文件和导出

- 函数文件名与主函数名一致，公共物理操作优先写成可单测函数，不把关键逻辑藏在一次性脚本中。
- 使用 `fullfile` 和调用者传入的路径，不硬编码个人机器目录。
- HDF5记录必要的参数、单位、随机种子、场景/回合编号、目标平面和模拟器状态。
- MATLAB生成物写入已忽略的 `data/` 或 `outputs/`，不提交大型二进制文件。
- 演示图和GIF必须标注是否有补偿；视觉演示不得被描述为算法比较结果。

## 验证

修改S0静态物理模块后运行：

```matlab
results = runtests('matlab/tests/AoS0PhysicsTest.m');
assertSuccess(results)
```

修改冻结流、时间推进或跨语言实现后运行：

```matlab
results = runtests('matlab/tests/AoS1DynamicsTest.m');
assertSuccess(results)
```

同时运行MATLAB代码分析。若改变MATLAB与PyTorch共享的公式，还必须重新生成参考HDF5并运行：

```powershell
.\.venv\Scripts\python.exe scripts\verify_s1_matlab_parity.py
```
