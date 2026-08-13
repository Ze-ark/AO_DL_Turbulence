# 动态仿真目录规则

本文件适用于 `src/simulation/`，并补充仓库根 `AGENTS.md`。

## 环境契约

- 环境必须保持可检查的 `reset(seed)` 和 `step(action)` 语义：当前观测与实际动作产生当前奖励和下一观测。
- 区分策略请求动作、经过模态限制后的目标动作和SLM实际加载动作；延迟、量化、饱和和变化率限制不得被策略绕过。
- 仿真真值不得无意泄漏给正式策略。使用真值模态时明确标注为oracle/诊断观测，并与未来ResUNet或全息重建观测分开。
- 高维oracle诊断必须通过`oracle_disturbance_modal()`直接投影湍流真值；不得用“残余模态减已施加模态”恢复真值，以免float32消差破坏严格对齐检查。
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

数据量诊断发现3600步附近仍可能改善后，128/256维收敛复查由用户运行：

```powershell
.\.venv\Scripts\python.exe scripts\train_s3_convergence_capacity.py --config configs\experiments\s3_convergence_capacity_v1.yaml
```

复查必须核对上游摘要与开发HDF5哈希，最低训练步数后才能早停，并分别保存每种未见风速/风向的结果。它仍是开发性诊断，不得读取原封存轨迹或直接授权RL。

收敛复查支持256维容量但仍未超过岭回归后，S3-B复杂动态门槛由用户运行：

```powershell
.\.venv\Scripts\python.exe scripts\train_s3_complex_dynamics.py --config configs\experiments\s3_complex_dynamics_v1.yaml
```

S3-B必须使用全新完整回合种子、两帧预测提前量和预声明的三类动态；带噪历史不能污染无噪未来目标。主线性基线按动态类型分别拟合并与GRU共享8帧历史。统计以模型初始化种子为主要复现单位，训练入口不得生成原S3或S3-B封存数据，也不得在门槛失败时自动进入RL。

S3-B门槛失败后，S4-A线性闭环鲁棒性比较由用户在IDE运行：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_linear_robustness.py
```

助手只允许运行 `--preflight-only` 和明确标注的 `--quick` CUDA冒烟。正式入口只能使用S3-B `train` 组拟合岭回归；传统控制器参数使用S4调参条件，控制器选择与开发门槛使用独立S4开发条件，封存条件不得生成或评估。所有控制器共享观测噪声、两帧延迟、动作限制和回合种子；科学指标使用无噪环境真值。若所有外推参数都超出安全阈值，必须保留此事实并选择最低违规候选继续诊断，不能隐藏违规或放宽门槛。

修改S4-A控制器、噪声观测、门槛或入口时至少运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_s2_controllers.py tests\test_s4_robust_control.py
.\.venv\Scripts\python.exe scripts\run_s4_linear_robustness.py --preflight-only
.\.venv\Scripts\python.exe scripts\run_s4_linear_robustness.py --quick
```

快速结果不得用于算法排名。正式结果完成后等待用户通知，再只读审计；不得自动打开S4封存条件或发送真实SLM动作。

S4-A开发门槛通过并完成审计后，S4-B只能运行冻结的泄漏积分器和不校正参照：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_sealed_validation.py --preflight-only
.\.venv\Scripts\python.exe scripts\run_s4_sealed_validation.py --acknowledge-sealed-test
```

第一条命令可以由助手运行，且不得生成封存数据。第二条命令必须由用户在IDE显式运行一次；源码哈希、控制器参数、条件顺序或上游哈希任一变化时必须在生成封存轨迹前失败。打开标记或输出目录存在时不得自动重试、删除或覆盖。封存结果完成后只读审计，不得因FAIL而修改参数重跑。

S4-C0完成并暴露慢稳定和严重配准失效边界后，SLM不可用期间进入S4-D0纯仿真鲁棒传统控制器开发：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_robust_controller_development.py --preflight-only
.\.venv\Scripts\python.exe scripts\run_s4_robust_controller_development.py --quick
.\.venv\Scripts\python.exe scripts\run_s4_robust_controller_development.py
```

助手只运行前两条；第三条由用户在IDE运行。S4-D0必须使用与S4-C0隔离的新完整回合种子，保留不校正和S4-B冻结积分器，并预声明候选选择规则。快速冒烟不得排名。S4-D0通过后仍需建立全新未见种子的S4-D1，S4-D1通过前不得创建或启动残差SAC正式训练。

S4-D0已完成审计并冻结 `tracking_conservative` 后，S4-D1只允许比较不校正、旧冻结积分器和该唯一冻结候选：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_robust_controller_validation.py --preflight-only
.\.venv\Scripts\python.exe scripts\run_s4_robust_controller_validation.py --acknowledge-unseen-validation
```

第一条可由助手运行，且必须保持未见轨迹未生成。第二条只能由用户在IDE运行一次；运行前写入 `UNSEEN_VALIDATION_OPENED.json`，输出目录存在时必须拒绝再次访问。S4-D1不得提供快速模式、候选排名或自动调参；源码、D0摘要、D0审计记录、D0轨迹清单、门槛、控制器参数或种子隔离任一不匹配时，必须在生成第一条轨迹前失败。正式结果完成后等待用户通知并只读审计，PASS前不得建立残差SAC，FAIL也不得删除输出后重跑。

S4-D1通过只读审计后，S4-D2残差SAC入口为：

```powershell
.\.venv\Scripts\python.exe scripts\train_s4_residual_sac.py --config configs\experiments\s4_residual_sac_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\train_s4_residual_sac.py --config configs\experiments\s4_residual_sac_v1.yaml --quick
.\.venv\Scripts\python.exe scripts\train_s4_residual_sac.py --config configs\experiments\s4_residual_sac_v1.yaml
```

助手只运行前两条；第三条正式训练由用户在IDE启动。S4-D2必须验证上游D1哈希但不得读取D1轨迹作为经验，训练、开发验证和未来D3种子必须隔离。策略不得读取观测末尾的仿真真值质量指标；零残差必须逐帧退化为冻结控制器；最终组合动作不能扩大0.15 rad单步范围。快速冒烟不排名，正式训练完成后等待用户通知并只读审计，不得自动创建或打开S4-D3。

S4-D2正式训练门槛失败后，只读失败诊断入口为：

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_s4_residual_sac.py --config configs\experiments\s4_residual_sac_diagnostic_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\diagnose_s4_residual_sac.py --config configs\experiments\s4_residual_sac_diagnostic_v1.yaml --quick
.\.venv\Scripts\python.exe scripts\diagnose_s4_residual_sac.py --config configs\experiments\s4_residual_sac_diagnostic_v1.yaml
```

助手只运行前两条；第三条正式诊断由用户在IDE启动。诊断只能只读加载冻结的最佳检查点，比较零、缩小、原始和反向残差，不得执行优化器更新、改写检查点、读取S4-D1轨迹或生成S4-D3轨迹。反向残差是事后故障定位，不是候选算法。正式结果完成后等待用户通知并只读审计，未经新实验设计不得自动重训。

失败诊断完成并由用户明确授权后，S4-D2-R1入口为：

```powershell
.\.venv\Scripts\python.exe scripts\train_s4_r1_residual_sac.py --config configs\experiments\s4_residual_sac_r1_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\train_s4_r1_residual_sac.py --config configs\experiments\s4_residual_sac_r1_v1.yaml --quick
.\.venv\Scripts\python.exe scripts\train_s4_r1_residual_sac.py --config configs\experiments\s4_residual_sac_r1_v1.yaml
```

助手只运行前两条；第三条正式训练由用户在IDE启动。R1唯一允许的科学变更是把残差物理上限从0.05 rad缩小到0.0125 rad；算法、网络、奖励、传统控制器、训练预算、动态条件、硬件档位和2%门槛必须保持原值。R1使用全新训练和开发种子，不读取D1、D2或诊断轨迹。R1失败后不得自动重训、换算法或降低门槛，下一步先建立同动作约束下的理想控制上限。

S4-D2-R1正式门槛失败并完成只读审计后，受限理想控制能力上限入口为：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_r1_oracle_bound.py --config configs\experiments\s4_r1_oracle_bound_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\run_s4_r1_oracle_bound.py --config configs\experiments\s4_r1_oracle_bound_v1.yaml --quick
.\.venv\Scripts\python.exe scripts\run_s4_r1_oracle_bound.py --config configs\experiments\s4_r1_oracle_bound_v1.yaml
```

助手只运行前两条；第三条正式CUDA仿真由用户在IDE启动。该阶段不训练模型，
但会显式读取当前与未来仿真真值，并逐回合选择最佳预见时域，所以只能标记为
不可部署的乐观能力诊断。它必须保留R1的±0.0125 rad残差、0.15 rad最终步长、
冻结传统控制器、硬件误差链和2%门槛，使用2500000段全新种子，禁止读取旧
轨迹、S4-D3和真实硬件。无论结果PASS或FAIL，都先等待用户通知并只读审计，
不得自动启动R2或打开S4-D3。

受限理想上限正式FAIL并审计后，非学习动作范围扫描入口为：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_r1_action_range_sweep.py --config configs\experiments\s4_r1_action_range_sweep_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\run_s4_r1_action_range_sweep.py --config configs\experiments\s4_r1_action_range_sweep_v1.yaml --quick
.\.venv\Scripts\python.exe scripts\run_s4_r1_action_range_sweep.py --config configs\experiments\s4_r1_action_range_sweep_v1.yaml
```

助手只运行前两条；第三条正式CUDA扫描由用户在IDE启动。扫描必须锁定上游
FAIL摘要、审计记录和源码哈希，使用2600000段全新完整回合种子，同时包含
R1的0.0125 rad和原D2的0.05 rad锚点，保留0.15 rad最终动作步长、硬件误差链
和原完整2%门槛。逐回合预见与动作范围包络只能标记为不可部署的乐观诊断。
正式输出完成后等待用户通知并只读审计；找到通过范围也不自动授权R2或S4-D3。

动作范围扫描正式FAIL并审计后，配准感知理想控制诊断入口为：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_registration_oracle.py --config configs\experiments\s4_registration_oracle_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\run_s4_registration_oracle.py --config configs\experiments\s4_registration_oracle_v1.yaml --quick
.\.venv\Scripts\python.exe scripts\run_s4_registration_oracle.py --config configs\experiments\s4_registration_oracle_v1.yaml
```

助手只运行前两条；第三条正式CUDA诊断由用户在IDE启动。该阶段只允许在固定
±0.05 rad残差、0.15 rad最终步长、相同未来真值、硬件链和原门槛下，比较
现有盲配准目标与使用精确仿真相位比例、平移和旋转的全瞳面最小二乘逆映射。
没有配准的档位必须保持数值不变。快速或正式运行崩溃后保留输出且不得自动
重试；PASS也只说明动作参数化值得设计，不授权R2、S4-D3或真实硬件。

配准感知理想控制正式FAIL并完成审计后，该阶段视为已封存。其正式入口在
源码或上游证据变化后必须拒绝重跑，不得用修改后的代码覆盖原结果。后续动作
表示容量扫描入口为：

```powershell
.\.venv\Scripts\python.exe scripts\run_s4_representation_capacity.py --config configs\experiments\s4_representation_capacity_v1.yaml --preflight-only
.\.venv\Scripts\python.exe scripts\run_s4_representation_capacity.py --config configs\experiments\s4_representation_capacity_v1.yaml --quick
.\.venv\Scripts\python.exe scripts\run_s4_representation_capacity.py --config configs\experiments\s4_representation_capacity_v1.yaml
```

助手只运行前两条；第三条正式CUDA扫描由用户在IDE启动。10、21、36和256维
表示必须保留完全相同的前10个Zernike锚点，并同时执行逐分量限制和旧10维
动作对应的总相位均方根预算，禁止因维数增加而扩大总动作能量。该阶段允许未来
仿真真值和精确配准参数，只能标记为不可部署的容量诊断。正式结果完成后等待
用户通知并只读审计；PASS也不授权R2、S4-D3或真实硬件。

改变公共接口、配置或数据结构时再运行全套：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

改变冻结流公式时还要运行MATLAB–PyTorch对照；改变GPU设备逻辑时运行CUDA冒烟。不得为了让测试通过而放宽CUDA硬门、删除物理断言或扩大科学结论。
