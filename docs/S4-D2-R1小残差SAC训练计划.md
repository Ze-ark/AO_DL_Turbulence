# S4-D2-R1小残差SAC训练计划

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `build / run-preparation`
- Origin Date: `2026-08-06`
- Verification Status: `READY_FOR_PREFLIGHT_AND_QUICK_SMOKE`
- Version Label: `s4d2_r1_small_residual_sac_v1`

## 一句话说明

R1重新训练残差SAC，但只把RL能够追加的最大物理动作从±0.05 rad缩小到
±0.0125 rad。它检验“原模型是不是因为动作太猛而失败”，不是换算法，也不是
放宽及格线。

## 结果状态

正式训练已经完成并审计为 **FAIL**：三个策略种子的平均桶内功率变化为
+0.0427%，0/3策略种子及0/18档位组合通过原门槛。下一步不再训练R1，而是
进入[S4-D2-R1理想控制上限计划](S4-D2-R1理想控制上限计划.md)。

## 为什么可以做R1

正式失败诊断显示：原100%残差使桶内功率下降0.3225%，25%残差平均提高
0.1005%，反向残差下降4.0526%。这支持继续检验小动作范围，但25%的事后结果
仍远低于2%门槛，不能直接成为新控制器。

R1由用户明确授权。程序会核对原训练、失败诊断及两个审计记录的哈希；任何
上游证据漂移都会在创建输出和占用CUDA前停止。

## 唯一改变的参数

| 项目 | 原S4-D2 | S4-D2-R1 |
|---|---:|---:|
| RL残差物理上限 | ±0.05 rad | **±0.0125 rad** |

以下内容全部不变：

- 算法仍为残差Soft Actor-Critic（SAC）；
- 策略输入100维、输出10维、隐藏层256×256；
- 最近4帧观测历史和冻结传统控制器状态；
- 桶内功率、归一化动作和违规奖励权重；
- 冻结的 `tracking_conservative` 参数；
- 3个策略种子、每种子499200条转移和59904次左右更新；
- 训练动态、硬件误差档位和开发验证门槛；
- 相对功率至少提高2%，Strehl不下降、相位误差不增加、违规率不超过5%。

归一化动作惩罚保持原值，所以R1检验的是更保守的物理动作范围。它并不精确
等价于诊断时把已训练动作事后乘0.25。

## 新种子隔离

| 用途 | 种子段 |
|---|---|
| R1三个正式训练策略 | 2300000、2320000、2340000起始 |
| R1开发验证 | 2400000、2401000、2402000起始 |
| R1快速冒烟 | 2460000和2475000段 |
| 未来理想控制上限 | 2500000段起，R1禁止访问 |

这些种子不与S4-D1、原S4-D2、失败诊断或S4-D3重叠。R1也不会读取旧轨迹作为
训练经验。

## 运行顺序

助手只运行预检和CUDA快速冒烟：

```powershell
.\.venv\Scripts\python.exe scripts\train_s4_r1_residual_sac.py --config configs\experiments\s4_residual_sac_r1_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\train_s4_r1_residual_sac.py --config configs\experiments\s4_residual_sac_r1_v1.yaml --quick
```

快速结果只证明软件链可以工作。正式训练由用户在IDE集成终端启动：

```powershell
.\.venv\Scripts\python.exe scripts\train_s4_r1_residual_sac.py --config configs\experiments\s4_residual_sac_r1_v1.yaml
```

程序先显示明确的 `S4-D2-R1`种子提示。底层复用原S4-D2训练内核，因此动态
进度条中的短标签仍可能显示“S4-D2策略种子”；应以R1提示、配置和
`outputs/s4_residual_sac_r1_v1`输出目录为准。进度条持续显示奖励、功率、违规、
策略损失、评价损失、预计剩余时间和CUDA显存。

完成后通知助手“`S4-D2-R1训练完成`”。不要自行重跑、改变奖励、降低门槛或
打开S4-D3。

## R1通过与失败

R1只有在三个策略种子的开发审计支持原2%门槛，并且功率、Strehl、相位误差和
违规率同时满足要求时，才有资格讨论下一步。训练摘要中的自动门槛仍需只读
审计，不能直接授权S4-D3。

如果R1失败：

1. 封存R1负结果，不自动重试；
2. 不马上切换TD3、PPO或其他RL算法；
3. 先在相同±0.0125 rad约束下建立理想控制上限；
4. 只有理想上限明显超过2%，才设计一次针对性的算法R2。

## 证据边界

R1仍是纯CUDA仿真。它不访问真实SLM，不代表中科微星
`FSLM-2K73-P04`性能，也不允许把快速冒烟、最佳训练点或单个策略种子写成正式
算法优越性结论。
