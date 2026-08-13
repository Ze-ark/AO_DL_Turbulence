# S4-D纯仿真鲁棒控制与RL路线

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `build / run-preparation`
- Origin Date: `2026-08-04`
- Verification Status: `S4_D1_ANALYZED_PASS_S4_D2_IN_PREPARATION`
- Version Label: `s4d_software_first_v3`

## 小白版说明

SLM不在身边不会阻塞软件研究。现在先让传统控制器学会处理“SLM动作慢”和“实际动作跟请求不一致”，再让强化学习只负责传统控制器没处理好的小部分。

```text
S4-C0失效边界
→ S4-D0开发鲁棒传统控制器
→ S4-D1用全新未见种子验证
→ S4-D2训练残差SAC
→ S4-D3比较RL与最强传统控制器
→ 等SLM可用后再做H1和S4-C1
```

## S4-D0已经得到什么

普通泄漏积分器只记得“自己请求了什么”。新增的跟踪泄漏积分器还读取观测中的“实际加载动作”，用请求与实际动作的差值做抗积分饱和反馈，并限制单帧请求变化。

S4-D0比较对象共6个：

1. 不校正；
2. S4-B冻结泄漏积分器；
3. 4组跟踪泄漏积分器候选。

它使用7个硬件档位、3种物理动态、每种32个完整回合和200帧。正式比较共25200个CUDA批处理时间步，生成126个HDF5轨迹。开发种子从 `1500000` 开始，与S4-C0完全隔离。

正式结果已经完成只读审计：4个候选全部通过6个必过档位，最终按“最差
档位增益最高”冻结 `tracking_conservative`。50%慢响应违规率由旧控制器
的8.34%降至0.79%，严重配准下相对不校正增益达到17.39%。详细证据见
[S4-D0鲁棒控制器开发审计记录](S4-D0鲁棒控制器开发审计记录.md)。

## 选择门槛

候选必须同时满足：

- 基准、三帧延迟、50%稳定速度、中等配准、严重配准和组合中等6个档位全部通过；
- 每个档位相对不校正的桶内功率平均增益至少10%；
- 桶内功率配对差值95%区间下界大于0.05；
- Strehl改善、残余相位误差下降；
- 总体和每种物理条件的违规率均不超过5%；
- 基准档位相对S4-B冻结控制器的功率下降不超过3%。

符合门槛后，选择“最差硬件档位仍保持最高增益”的候选。没有候选全部通过时，结果必须为FAIL，不能临时放宽门槛。

## S4-D1验证什么

S4-D1不再比较4个候选，也不允许重新选型。它只运行：

1. 不校正参照；
2. S4-B冻结泄漏积分器；
3. 参数完全冻结的 `tracking_conservative`。

使用与S4-D0相同的7个硬件档位和6个必过门槛，但换成3种未用于选型的
物理参数与96个全新完整回合。PASS必须同时满足：

- 冻结控制器在6个必过档位全部通过原门槛；
- 基准功率相对旧控制器下降不超过3%；
- 50%慢响应违规率不超过5%，并且低于旧控制器；
- 严重配准下，相对旧控制器的桶内功率配对差值95%区间下界大于0；
- 不允许看完结果后换候选、改参数或重跑。

## 运行方法

S4-D0正式开发比较已经运行并封存，不得再次执行。S4-D1先做只读预检：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_robust_controller_validation.py --preflight-only
```

正式未见验证只能由用户在IDE集成终端显式运行一次：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_robust_controller_validation.py --acknowledge-unseen-validation
```

运行完成后停止并通知助手“`S4-D1运行完成`”。不要自行重跑，也不要直接
开始强化学习。

## RL何时开始

S4-D1正式结果已经通过只读审计，现在实现残差SAC：

```text
最终动作 = 冻结的鲁棒传统控制器动作 + RL小幅修正动作
```

此前失败的GRU和S3-B结果继续保留。残差SAC是新的模型自由探索路线，不能写成GRU或PO4AO门槛已经通过。RL必须与S4-D1冻结的最强传统控制器共享观测、硬件误差、动作限制、训练预算和测试轨迹。

S4-D2只使用全新训练和开发验证种子。策略读取最近4帧残余模态和实际动作、当前请求以及冻结控制器动作，不读取仿真真值质量指标。RL残差限制为每模态±0.05 rad，合并后的最终动作仍受原±0.15 rad单步范围约束。正式训练由用户在IDE启动，详见 [S4-D2残差SAC训练计划](S4-D2残差SAC训练计划.md)。

## 当前状态

```text
S4-C0：COMPLETED / ANALYZED / FAIL
S4-D0：COMPLETED / ANALYZED / PASS
S4-D1：COMPLETED / ANALYZED / PASS / SEALED FROM RL
Residual SAC：S4-D2 RUN PREPARATION
H1 / S4-C1：DEFERRED UNTIL SLM AVAILABLE
真实SLM动作：NOT AUTHORIZED
```
