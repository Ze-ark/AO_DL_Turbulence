# S4-D2残差SAC失败诊断计划

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `build / run-preparation`
- Origin Date: `2026-08-06`
- Verification Status: `ANALYZED`
- Version Label: `s4d2_residual_sac_diagnostic_v1`

## 一句话说明

这一步不再训练模型，只把已经训练好的SAC动作分别乘以0、0.25、0.5、1和
-1，在完全相同的动态湍流中重放。目的是查清模型为什么让补偿效果变差。

## 为什么现在不继续加训练量

三个独立模型和全部18个“模型×环境”组合都变差，说明结果不像一次偶然波动。
直接增加训练量既费时间，也无法区分程序接错、动作太大、方向学反或奖励设计
不合适。先做缩放诊断成本更低，结论也更容易解释。

## 五种动作是什么意思

| 名称 | 做法 | 要回答的问题 |
|---|---|---|
| 零残差 | SAC动作乘0 | 程序能否准确退化为原传统控制器？ |
| 四分之一残差 | SAC动作乘0.25 | 动作缩小后是否少伤害？ |
| 二分之一残差 | SAC动作乘0.5 | 伤害是否随动作大小变化？ |
| 原始残差 | SAC动作乘1 | 在新种子上是否仍复现负结果？ |
| 反向残差 | SAC动作乘-1 | SAC是否可能把方向学反？ |

反向残差只是“把箭头倒过来试一下”的故障检查，不是新控制器，也不能拿去做
论文算法排名。

## 公平比较

- 三个最佳检查点保持只读，不做任何优化器更新；
- 每种动作使用相同的湍流种子、观测噪声和200帧长度；
- 使用三个全新的动态条件和四个硬件误差档位；
- 正式诊断共使用2200000段的新种子，不碰S4-D1、S4-D2训练/验证和S4-D3；
- 真值桶内功率、Strehl、相位均方根误差和违规率与训练式奖励分开保存；
- 全过程仅为CUDA纯仿真，不发送SLM图案。

## 运行顺序

助手先运行只读预检和CUDA快速诊断：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_s4_residual_sac.py --config configs\experiments\s4_residual_sac_diagnostic_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\diagnose_s4_residual_sac.py --config configs\experiments\s4_residual_sac_diagnostic_v1.yaml --quick
```

两项通过后，正式诊断由用户在IDE集成终端运行：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_s4_residual_sac.py --config configs\experiments\s4_residual_sac_diagnostic_v1.yaml
```

这条命令不会训练。进度条显示已完成轨迹组、最近功率差、预计剩余时间和CUDA
显存。完成后通知助手“`S4-D2诊断完成`”，不要自行重跑或打开S4-D3。

## 结果怎么读

| 看到的结果 | 小白解释 | 下一步候选方向 |
|---|---|---|
| 零残差不等于传统基线 | 程序接线可能有问题 | 先修接线，现有RL结论暂不解释 |
| 0.25和0.5明显好于1 | SAC补充动作可能过猛 | 再研究更小动作范围或动作正则 |
| -1优于1且相对基线为正 | SAC可能系统性学反方向 | 检查动作符号、时延和奖励归因 |
| 诊断奖励最好者不是真值功率最好者 | 训练目标可能带偏模型 | 重审奖励、测量噪声和违规权重 |
| 上述现象都不明显 | 不能用简单原因解释 | 检查状态信息、信用分配或停止RL路线 |

这些只叫“支持某种原因”，不叫“证明根因”。如果要改变奖励、动作范围或重新
训练，必须先审计正式诊断结果并建立新实验版本。

## 输出文件

- `summary.json`：总结果和四类保守诊断标签；
- `scenario_records.csv`：每个策略、动作缩放、硬件档位和动态条件的均值；
- `preflight.json`：上游失败结论、检查点、源码和种子隔离证明；
- `source_manifest.json`：本次诊断代码哈希；
- `effective_config.json`：实际运行参数。

## 证据边界

快速诊断只验证软件能跑，不能解释失败原因。正式诊断也是事后开发分析，不能
授权S4-D3，不能声称RL优于传统控制器，也不代表两台
`FSLM-2K73-P04`的真实硬件表现。

## 正式诊断结果更新

正式诊断已完成并经只读审计。接线检查PASS；100%残差使桶内功率下降
0.3225%，25%残差平均提高0.1005%，反向100%残差下降4.0526%。结果支持
“动作幅度过大”，但25%分支未达到2%门槛。用户已明确授权建立只缩小残差物理
上限的 [S4-D2-R1小残差SAC训练计划](S4-D2-R1小残差SAC训练计划.md)。
