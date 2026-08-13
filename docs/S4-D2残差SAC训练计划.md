# S4-D2残差SAC训练计划

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `build / run-preparation`
- Origin Date: `2026-08-05`
- Verification Status: `ANALYZED_FAIL`
- Version Label: `s4d2_residual_sac_v1`

## 一句话说明

冻结已经通过S4-D1的 `tracking_conservative`，强化学习只学习10维小幅补充
动作。S4-D2只使用新的训练和开发验证回合，不读取S4-D1轨迹，也不打开未来
S4-D3封存测试。

## 每一帧发生什么

```text
最近4帧残余模态和实际动作
→ 冻结控制器给出基础动作
→ SAC给出不超过±0.05 rad的补充动作
→ 合并后仍限制为±0.15 rad/帧
→ SLM误差仿真和湍流推进一帧
→ 用带噪桶内功率、残差动作和违规计算训练奖励
→ 保存转移到经验池并更新SAC
```

策略不读取S1观测末尾的仿真真值Strehl和真值桶内功率。Strehl、真值桶内
功率和残余相位RMSE只用于独立科学评价，避免策略偷看模拟器答案。

## 网络与预算

| 项目 | 冻结值 |
|---|---:|
| 策略输入 | 100维 |
| 策略输出 | 10维归一化残差 |
| 策略和评价网络隐藏层 | 256、256 |
| 并行环境 | 32 |
| 回合长度 | 200帧 |
| 正式策略种子 | 3个 |
| 每种子训练量 | 499200条转移，正好78批完整回合 |
| 预热经验 | 20000条转移 |
| 经验池 | 500000条 |
| 小批量 | 256 |

500000是近似目标；实际冻结为499200，目的是不在回合中间截断训练数据。

## 数据隔离

- S4-D1的96个回合和1600000种子段只做完整性门槛核验；
- S4-D2训练使用1700000、1720000和1740000起始的隔离种子段；
- S4-D2开发验证使用1800000段；
- 2100000段只为未来S4-D3预留，本入口禁止访问；
- 按完整回合保存和比较，不随机拆散相邻帧作为独立测试样本。

## 训练奖励与科学指标

训练奖励为：

```text
带噪桶内功率 - 小幅残差动作代价 - 违规代价
```

开发验证仍分别报告真值桶内功率、Strehl、残余相位RMSE、违规率和配对95%
区间。奖励升高但独立科学指标没有改善时，门槛判定为FAIL。

建议的开发门槛是：相对冻结传统控制器桶内功率至少提高2%，功率差区间下界
大于0，Strehl不下降、相位RMSE不增加且违规率不超过5%。它是进入S4-D3前
需要审计的开发门槛，不是已经得到的结果。

## 运行顺序

助手可以运行只读预检和明确标注的CUDA快速冒烟：

```powershell
.\.venv\Scripts\python.exe scripts\train_s4_residual_sac.py --config configs\experiments\s4_residual_sac_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\train_s4_residual_sac.py --config configs\experiments\s4_residual_sac_v1.yaml --quick
```

快速结果只证明软件链可运行。正式训练必须由用户在IDE集成终端启动：

```powershell
.\.venv\Scripts\python.exe scripts\train_s4_residual_sac.py --config configs\experiments\s4_residual_sac_v1.yaml
```

程序持续显示总转移数、平均奖励、带噪功率、违规率、策略损失、评价损失、
预计剩余时间和CUDA显存，并写入CSV、JSONL、周期检查点、最佳检查点和最终
摘要。训练结束后停止并通知助手“`S4-D2训练完成`”，不要自行打开S4-D3。

## 证据边界

S4-D2是纯仿真RL训练。即使开发验证PASS，也不能声称RL已经通过封存比较，
更不能声称完成真实SLM闭环。S4-D3必须使用新种子做一次冻结比较；真实硬件
仍需等待H1标定和影子测试。

## 正式训练结果更新

正式训练已经完成。三个策略种子相对冻结传统控制器的桶内功率平均变化为
-0.4409%，三个种子均为负，开发门槛通过数为0/3。因此S4-D3保持关闭，原
训练入口不得重跑。完整证据见 [S4-D2残差SAC训练审计记录](S4-D2残差SAC训练审计记录.md)。

当前下一步不是增加训练量，而是运行只读的残差缩放和方向诊断，详见
[S4-D2残差SAC失败诊断计划](S4-D2残差SAC失败诊断计划.md)。
