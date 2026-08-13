# S4-B一次性封存验证计划

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `plan + build`
- Origin Date: `2026-08-04`
- Verification Status: `PARTIALLY VERIFIED`
- Version Label: `ao_s4b_sealed_validation_v1`

## 先说结论

S4-B已经在2026-08-04运行一次并完成只读审计，最终门槛为：

```text
PASS
```

完整数值和审计边界见 [S4-B一次性封存验证记录](S4-B一次性封存验证记录.md)。历史运行规则保留在本文中，不能据此再次运行。

S4-A从开发条件中选出的唯一候选为：

```text
泄漏积分器：gain=0.35，leak=0.10
```

S4-B不再挑算法，也不再调参数。它只问：

> 这个已经冻结的控制器，在从未运行过的六个封存条件中，是否仍然有效且安全？

本轮仍是CUDA纯仿真，不是强化学习，不会向真实空间光调制器（SLM）发送图案。

## 为什么只能运行一次

封存条件相当于考试卷。如果看过结果后修改参数再考，测试集就变成了调参集，结果会失去可信度。

程序采取三层保护：

1. 必须显式加入 `--acknowledge-sealed-test`；
2. 第一条轨迹运行前写入 `SEALED_TEST_OPENED.json`；
3. 输出目录一旦存在，程序拒绝第二次运行。

如果运行中断，不要删除输出目录或自行重试。保留报错并先通知助手。

## 冻结内容

| 项目 | 冻结值 |
|---|---|
| 候选控制器 | 泄漏积分器 |
| 增益 | 0.35 |
| 泄漏 | 0.10 |
| 参照 | 不校正 |
| 主要指标 | 桶内功率 |
| 延迟 | 2帧 |
| 每个条件回合数 | 32 |
| 每个回合长度 | 200帧 |
| 封存条件 | 固定冻结流2组、带沸腾2组、组合复杂动态2组 |

源码清单包含运行入口、控制器、环境、湍流、光学传播、评价和CUDA设备代码的SHA-256哈希。任何一个文件改变，预检都会停止，不能静默更新后继续。

## 通过条件

冻结控制器必须同时满足：

1. 桶内功率配对差值的95%区间下界大于0.05；
2. 平均桶内功率相对增益至少10%；
3. Strehl配对差值的95%区间下界大于0；
4. 残余相位误差差值的95%区间上界小于0；
5. 总违规率不超过5%；
6. 六个条件的桶内功率全部提高；
7. 六个条件的违规率全部不超过5%。

其中前两项防止“数值虽然大于零，但实际改善太小”也被写成成功。这些阈值已在打开封存条件前写入配置。

## 运行方式

不会打开封存数据的预检已经完成，也可以自行复核：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_sealed_validation.py --preflight-only
```

正确输出应包含：

```text
READY_WITH_SEALED_TEST_STILL_CLOSED
sealed_test_accessed: false
source_manifest_verified: true
```

确认接受“一次性打开、不得调参、不得自动重跑”后，在IDE集成终端运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_sealed_validation.py --acknowledge-sealed-test
```

正式运行共执行2400个CUDA闭环步，并实时显示条件、控制器、进度、预计剩余时间和显存。

## 输出

```text
outputs/s4_sealed_validation_v1/SEALED_TEST_OPENED.json
outputs/s4_sealed_validation_v1/summary.json
outputs/s4_sealed_validation_v1/source_manifest.json
outputs/s4_sealed_validation_v1/controller_summary.csv
outputs/s4_sealed_validation_v1/condition_summary.csv
outputs/s4_sealed_validation_v1/trajectories/
```

终端出现下面提示后停止：

```text
S4-B一次性封存验证完成，请停止并告诉助手：S4-B运行完成。
```

助手会只读复算12个轨迹文件中的配对指标和门槛，不会自动进入S4-C、RL或真实SLM实验。

## 当前状态

- S4-A正式开发比较：已完成并审计，门槛PASS；
- 冻结控制器：已确认；
- 源码哈希清单：冻结清单中的文件均通过检查，但审计发现实际依赖的 `src/simulation/slm.py` 未被列入；
- CUDA预检：已通过；
- 不带确认参数的拒绝测试：已通过；
- S4-B封存条件：已打开并完成唯一一次正式运行，门槛PASS；
- S4-B输出：永久只读，不得删除标记、重新调参或重跑；
- 下一步：S4-C0保守硬件误差纯仿真，详见 [S4-C硬件误差纯仿真计划](S4-C硬件误差纯仿真计划.md)；
- 真实SLM动作：未执行。
