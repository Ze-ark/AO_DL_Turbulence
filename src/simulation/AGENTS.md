# 动态仿真目录规则

本文件适用于 `src/simulation/`，并补充仓库根 `AGENTS.md`。

## 环境契约

- 环境必须保持可检查的 `reset(seed)` 和 `step(action)` 语义：当前观测与实际动作产生当前奖励和下一观测。
- 区分策略请求动作、经过模态限制后的目标动作和SLM实际加载动作；延迟、量化、饱和和变化率限制不得被策略绕过。
- 仿真真值不得无意泄漏给正式策略。使用真值模态时明确标注为oracle/诊断观测，并与未来ResUNet或全息重建观测分开。
- 奖励可以用于训练，但信息字典必须保留独立的Strehl、桶内功率、相位RMSE、动作代价和违规信息。
- 批量环境中的每个回合使用独立、可追溯的随机种子。

## 时间湍流与数值安全

- 严格泰勒冻结流和带沸腾模型由显式参数区分；风速、方向、时间步长和空间采样单位必须一致。
- 周期相位屏不得在单个回合内完整绕回并重复同一轨迹；修改网格、风速或回合长度时保留对应防护测试。
- PyTorch相位屏必须保持Von Karman结构函数统计测试；修改傅里叶移位时保留整数平移、亚像素可逆性和MATLAB回归。
- 正式动态环境使用CUDA。CPU只用于小尺寸、确定性的单元测试；不得把CPU测试结果报告成正式吞吐或性能结果。

## 变更验证

正式训练遵守根目录的“训练执行协作”：由用户在 IDE 中启动，助手只准备和验证入口。新增 S3 或后续训练脚本时必须复用 `src/training_progress.py`，并在训练期间持续写入损失历史；不得只在进程结束时打印一次结果。

至少运行直接相关测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_s1_turbulence.py tests\test_s1_env.py -q
```

修改S2控制器、配对统计或轨迹导出时运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_s2_controllers.py -q
.\.venv\Scripts\python.exe scripts\run_s2_baselines.py --quick
```

快速入口只证明软件链可运行；正式比较必须使用独立开发回合和封存测试回合，并把输出标记为纯仿真结果。

修改 S2 ResUNet 的观测、数据或模型契约时，还要运行迁移检查和相关单元测试；只有模型或数据契约确实改变时才重跑训练与昂贵的最终比较：

```powershell
.\.venv\Scripts\python.exe scripts\check_s2_resunet_transfer.py
.\.venv\Scripts\python.exe scripts\train_s2_dynamic_resunet.py
.\.venv\Scripts\python.exe scripts\run_s2_baselines.py --final
```

确定性复现验证要求对同一最终命令做两次运行，使用 `scripts/verify_s2_final_reproducibility.py snapshot` 和 `compare` 比较非计时指标及轨迹哈希。

S3 动力学训练只能由用户在 IDE 启动：

```powershell
.\.venv\Scripts\python.exe scripts\train_s3_dynamics.py --config configs\experiments\s3_dynamics_v1.yaml
```

训练入口只允许校验 `sealed_test_conditions` 元数据不重叠，不得生成或评估封存回合。封存评估必须检查验证门槛、配置哈希、环境哈希和检查点哈希，并要求显式 `--acknowledge-sealed-test`；没有这些条件时应在生成测试数据前失败。

原 S3 门槛失败后的数据量诊断由用户运行：

```powershell
.\.venv\Scripts\python.exe scripts\train_s3_data_scaling_diagnostic.py --config configs\experiments\s3_data_scaling_diagnostic_v1.yaml
```

诊断必须按完整回合、物理条件均衡抽样，并对不同数据规模固定优化器更新次数；不得读取原 S3 封存轨迹。诊断结论不能直接授权 RL 或原封存评估。

改变公共接口、配置或数据结构时再运行全套：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

改变冻结流公式时还要运行MATLAB–PyTorch对照；改变GPU设备逻辑时运行CUDA冒烟。不得为了让测试通过而放宽CUDA硬门、删除物理断言或扩大科学结论。
