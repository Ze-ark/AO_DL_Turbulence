# Phase 2 文献调查：强化学习自适应光学（入门版）

> 状态：Phase 2 Investigation 已完成
> 检索截止日期：2026-07-22
> 本阶段任务：查清已有工作、核验论文、标记证据强弱。
> 本阶段不做的事：不宣布“首创”，不最终确定创新点，也不开始写强化学习代码。

## 1. 三分钟读懂本阶段结果

先说最重要的结论：原蓝图中的一些想法已经有人做过，不能单独当创新点。

1. **“传统积分控制器 + 强化学习修正”已经做过。** Pou 等人在 2022 年的多智能体 SAC 工作中，把强化学习动作加在积分器命令之上；2024 年又把监督学习重构器、积分器和强化学习预测控制组合起来。因此，“残差 SAC”这四个字本身不是新意。
2. **只用图像、不直接测完整波前的强化学习也已经做过。** 已有研究使用焦平面图像、图像质量分数或低维光电探测器控制可变形镜。
3. **监督学习 + 强化学习也已经做过。** 2024 年已有论文先用监督学习解决金字塔波前传感器的非线性重构，再用在线强化学习做预测控制。
4. **动作平滑或动作正则化也已经进入 AO 强化学习。** 2026 年已有针对光卫星通信 AO 的动作正则化强化学习论文。
5. **强化学习 AO 已经走出纯仿真。** 证据已经覆盖显微镜和天文实验台；2026 年 7 月的 A&A 期刊论文还报告了 PO4AO 的真实天文观测闭环测试。
6. **SAC 不是在所有 AO 场景中都最好。** 一项无波前传感器的卫星通信仿真中，PPO 优于同场比较的 SAC 和 DDPG。这只能说明算法优劣依赖具体观测、动作、奖励和动力学，不能直接推导出“PPO 永远更好”。
7. **本轮没有找到“离轴数字全息复场观测 + 强化学习 AO 闭环控制”的直接同题论文。** 找到了全息波前传感器的闭环 AO 研究，但它不是强化学习。这是一个值得 Phase 3 继续核查的候选空缺，不是“全球首次”的证明。

对当前项目最直接的提醒是：

> 下一步不能只说“我们用 SAC”“我们加入历史帧”或“我们让 RL 修正积分器”。真正可能形成贡献的部分，必须落在项目特有的可部署观测、物理闭环、约束与时延、严格对照，或者全息测量与控制的结合方式上。

## 2. 这次文献调查想回答什么

本轮围绕五个简单问题展开：

1. 强化学习是否已经用于自适应光学闭环控制？
2. 已有算法每一步看什么、控制什么、怎样得到分数？
3. “传统控制器 + 强化学习微调”是否已有先例？
4. 焦平面图像、无波前传感器观测和全息观测分别发展到哪一步？
5. 当前项目至少要与哪些强基准比较，才能说明强化学习确实有价值？

## 3. 检索方法

### 3.1 检索范围

- 主时间范围：2015–2026 年。
- 为建立强控制基线，保留 1993 年时延 LQG 和 2006 年 AO 最优控制等奠基论文。
- 数据源：OpenAlex、Crossref、Semantic Scholar、arXiv、PubMed/PMC，以及 Optica、A&A、SPIE、IEEE、PMLR 等出版社或会议的正式页面。
- 只把论文、正式会议论文或可核验预印本作为证据；搜索摘要网站和二手解读仅用于找到原文，不作为结论依据。

### 3.2 主要检索词

使用过的核心查询包括：

```text
adaptive optics reinforcement learning
wavefront sensorless adaptive optics deep reinforcement learning
model-based reinforcement learning adaptive optics
predictive control adaptive optics machine learning
focal plane wavefront correction reinforcement learning
off-axis holography adaptive optics reinforcement learning
holographic wavefront sensing reinforcement learning adaptive optics
adaptive optics residual reinforcement learning integrator
adaptive optics action regularization safe reinforcement learning
on-sky reinforcement learning adaptive optics
```

### 3.3 筛选规则

纳入：

- 直接研究 AO 闭环强化学习控制；
- 能作为强对照的积分、LQG、MPC 或数据驱动预测控制；
- 能回答焦平面观测、部分可观测、时延、动作约束或全息测量问题的原始研究；
- 题名、作者、年份和 DOI 或 arXiv 编号能够核验。

排除：

- 与 AO 无关的普通光学强化学习；
- 只有静态重构、完全没有控制含义的论文，除非它直接决定本项目的观测设计；
- 已有正式期刊版时重复计入其 arXiv 或会议前身；
- 只有二手摘要、元数据不确定或正文证据不足的记录；
- 综述作为主要效果证据。

### 3.4 检索计数

- OpenAlex 四组宽检索各取前 25 条，共返回 100 条记录。
- 去重后为 67 条，其中 18 条从题名即可判断与 AO–RL 直接相关。
- 随后通过出版社定向检索、参考文献追踪和精确题名检索补充经典控制、焦平面观测、全息测量及 2026 年最新论文。
- 本报告重点注释 28 项来源：17 项直接 AO–RL 研究、6 项强控制基线、3 项通用 RL 方法依据、2 项观测或全息背景研究。

搜索网站的总命中数会随时间和排序变化，因此这里不伪造一个看似精确的“全互联网总命中数”。这是一项系统化的范围调查，而不是已经注册方案的正式系统综述。

## 4. 怎样理解证据等级

ARS 的通用证据体系会把这些论文归为单项工程或计算研究，整体属于 **Level VI**。这不是说论文差，而是说明它们不是多中心随机试验或汇总分析，不能把单篇结果当成普遍真理。

为了便于阅读，本报告再加一个工程证据等级：

| 等级 | 小白解释 |
|---|---|
| A | 有真实硬件、实验台或真实天文观测，方法和比较较完整 |
| B | 同行评审的仿真或计算研究，方法可信，但还缺真实部署证明 |
| C | 会议论文、最新预印本或摘要级证据，适合发现方向，不适合下强结论 |

等级评价的是“这篇论文对当前问题能证明多少”，不是给作者或期刊排名。

## 5. 直接 AO–RL 文献图谱

### 5.1 早期无波前传感器路线

#### 1. Hu 等，2018，IEEE Photonics Technology Letters

- 论文：[Build the Structure of WFSless AO System Through Deep Reinforcement Learning](https://doi.org/10.1109/LPT.2018.2874998)
- 做法：用 DDPG 从远场光强图中学习连续的可变形镜控制。
- 意义：证明“没有传统波前传感器，也可以把 AO 写成强化学习问题”并非新想法。
- 局限：纯仿真、相机和系统条件较理想，没有真实时延、漂移和硬件安全测试。
- 证据：B−。

#### 2. Hu 等，2019，Optik

- 论文：[Self-learning control for wavefront sensorless adaptive optics system through deep reinforcement learning](https://doi.org/10.1016/j.ijleo.2018.09.160)
- 做法：仍以 DDPG 从远场图像控制 59 单元可变形镜。
- 报告结果：在论文设置中，校正效果接近 SPGD 和一般优化方法，同时减少迭代次数。
- 局限：仍为理想化仿真；奖励的完整定义无法从本轮可访问正式预览中可靠核实，因此本报告不补写公式。
- 证据：B−。

### 5.2 使用 WFS 遥测的动态闭环路线

#### 3. Nousiainen 等，2021，Optics Express

- 论文：[Adaptive optics control using model-based reinforcement learning](https://doi.org/10.1364/OE.420270)
- 做法：先从遥测数据学习“系统下一步会怎样变化”，再利用这个学习到的模型选择可变形镜动作。
- 观测：最近若干帧 WFS 残差和历史控制命令。
- 动作：可变形镜的增量命令。
- 奖励：下一帧残差越小，奖励越高。
- 重要结果：仿真中可预测湍流，并能适应 WFS 与可变形镜之间的配准误差。
- 局限：模型预测控制每步计算约 80–120 ms，无法直接满足 kHz AO；仅有仿真证据。
- 证据：B+。

#### 4. Landman 等，2021，JATIS

- 论文：[Self-optimizing adaptive optics control with reinforcement learning for high-contrast imaging](https://doi.org/10.1117/1.JATIS.7.3.039002)
- 做法：使用带记忆的模型无关强化学习，让控制器从历史残差和历史动作中学习振动与风场规律。
- 动作：tip–tilt 或高阶可变形镜的增量命令。
- 验证：tip–tilt 有仿真和实验台；高阶镜结果主要是仿真。
- 局限：高阶系统缺少硬件验证，真实探索的安全与样本效率没有形式保证。
- 证据：A−。

#### 5. Pou 等，2022，Optics Express

- 论文：[Adaptive optics control with multi-agent model-free reinforcement learning](https://doi.org/10.1364/OE.444099)
- 做法：把高维镜面控制分给多个局部智能体，每个智能体使用 SAC；强化学习动作作为积分控制器上的附加修正。
- 关键控制式可简单理解为：

  > 新镜面命令 = 上一步命令 − 积分器修正 + 强化学习修正。

- 验证：8 m 望远镜、40×40 Shack–Hartmann WFS 的数值闭环。
- 报告结果：优于积分器，并接近拥有理想大气先验的 LQG。
- 局限：纯仿真；LQG 获得了很强的理想先验，比较条件并不等价；局部智能体划分依赖模态近似可分。
- 证据：B+。

#### 6. Nousiainen 等，2022，Astronomy & Astrophysics

- 论文：[Toward on-sky adaptive optics control using reinforcement learning](https://doi.org/10.1051/0004-6361/202243311)
- 做法：PO4AO 先学习系统动力学，再用学习到的模型训练一个执行很快的控制策略。
- 观测：WFS 残差历史和历史可变形镜动作。
- 验证：8 m/40 m 仿真和 MagAO-X 实验台。
- 报告结果：控制区日冕对比度提升约 3–5 倍，典型训练约 5–10 s，推理低于 1 ms。
- 局限：当时尚未完成真实天文观测；需要安全的初始数据采集和并行训练设施。
- 证据：A−。

#### 7. Nousiainen 等，2024，JATIS

- 论文：[Laboratory experiments of model-based reinforcement learning for adaptive optics control](https://doi.org/10.1117/1.JATIS.10.1.019001)
- 做法：在 ESO GHOST 实验台实现 PO4AO，并让训练和控制同时运行。
- 奖励：既惩罚波前残差，也惩罚过大的动作。
- 验证：低光通量、振动、控制时延和人为引入的配准误差。
- 工程结果：PyTorch 路径额外增加约 700 微秒延迟；长历史有助于学习 16 Hz 振动。
- 局限：仍是实验台；存在训练热身、CPU/GPU 搬运、时延抖动和部署成本。
- 证据：A。

#### 8. Pou 等，2024，Optics Express

- 论文：[Integrating supervised and reinforcement learning for predictive control with an unmodulated pyramid wavefront sensor for adaptive optics](https://doi.org/10.1364/OE.530254)
- 做法：离线监督学习负责非线性波前重构，在线 SAC 负责预测性残差控制，同时控制高阶镜和 tip–tilt 台。
- 重要重叠：它与本项目“保留现有 ResUNet，再让 RL 学习剩余控制”的设想非常接近。
- 验证：8 m、1 kHz、两帧延迟的高阶仿真。
- 局限：纯 frozen-flow 仿真；积分基线采用全局增益，不是最强的逐模态增益基线；尚无实机证据。
- 证据：B+。

#### 9. Nousiainen 等，2026，Astronomy & Astrophysics

- 论文：[On-sky demonstration of reinforcement learning for adaptive optics control](https://doi.org/10.1051/0004-6361/202659769)
- 做法：在 OHP 1.52 m 望远镜的 PAPYRUS 系统上运行 PO4AO，与标准积分器比较。
- 证据：多个夜晚、多个目标、不同通量和天气条件；论文报告所有测试配置中均优于所用积分器，并能学习振动。
- 工程问题：Python 实现额外增加约 720–750 微秒延迟，还出现时延抖动和偶发丢帧。
- 局限：单一望远镜和控制系统；积分器不是所有目标都重新做最强逐模态调参；部分 PSF 和遥测改善不完全一致。
- 版本核验：Crossref 显示它已在 2026 年 7 月正式成为 A&A 711, A74 期刊论文，不再只按 arXiv 预印本处理。
- 证据：A−。

### 5.3 图像、低维探测器和无 WFS 路线

#### 10. Durech 等，2021，Biomedical Optics Express

- 论文：[Wavefront sensor-less adaptive optics using deep reinforcement learning](https://doi.org/10.1364/BOE.427970)
- 做法：在荧光共焦显微镜中，用最近几步图像质量分数控制 5 个低阶 Zernike 模态。
- 验证：仿真训练、原位训练和真实显微镜实验。
- 报告结果：与 Zernike 爬山法相比，所需校正步骤减少约 6–7 倍；仿真策略可迁移到该实验系统。
- 局限：只有 5 个低阶模态，目标较静态，不能直接外推到快速高维大气 AO。
- 利益冲突提示：全文披露一名作者与 Seymour Vision 相关；这不代表结果无效，但应在引用时保留披露信息。
- 证据：A−。

#### 11. Parvizi 等，2023，Photonics

- 论文：[Reinforcement Learning Environment for Wavefront Sensorless Adaptive Optics in Single-Mode Fiber Coupled Optical Satellite Communications Downlinks](https://doi.org/10.3390/photonics10121371)
- 做法：用低维光电探测器反馈，比较 PPO、SAC 和 DDPG 控制光纤耦合。
- 结果边界：在该准静态仿真环境中，PPO 优于 SAC 和 DDPG，达到理想化 Shack–Hartmann 方案最大性能的约 86%。
- 局限：主要是准静态和半动态仿真；收敛所需动作数对真实快速湍流可能过多。
- 项目意义：不能仅凭“连续动作适合 SAC”就提前锁定算法，必须在本项目环境中公平比较。
- 证据：B。

#### 12. Xu 等，2024，Biomedical Optics Express

- 论文：[Image metric-based multi-observation single-step deep deterministic policy gradient for sensorless adaptive optics](https://doi.org/10.1364/BOE.528579)
- 做法：构造多组图像质量观测，并利用带记忆的 DDPG 减少无传感器 AO 的搜索步骤。
- 验证：仿真和原位迁移实验；论文报告相对 Zernike 模态爬山法显著减少迭代。
- 局限：观测采集本身需要额外探测动作与时间；与高速大气主环并不等价。
- 证据：A−。

#### 13. Gutierrez 等，2024，Optics Express

- 论文：[Image-based wavefront correction using model-free reinforcement learning](https://doi.org/10.1364/OE.529415)
- 做法：只使用一对带相位多样性的焦平面图像，让模型无关 RL 同时学习相位反演和镜面校正。
- 意义：“用焦平面图像直接做 RL 波前校正”已经存在，不能作为宽泛创新点。
- 局限：主要是仿真；多样性图像会增加观测时间，且真实系统的模型差异仍需实验验证。
- 证据：B+。

#### 14. Parvizi 等，2026，JOSA B

- 论文：[Action-Regularized Reinforcement Learning for Adaptive Optics in Optical Satellite Communication](https://doi.org/10.1364/JOSAB.578050)
- 做法：在无 WFS 的卫星通信 AO 中，对策略动作的变化进行状态自适应平滑约束。
- 意义：“惩罚镜面抖动、让动作更平滑”也已有直接 AO–RL 先例。
- 局限：主要是数值环境；动作平滑不等于每一步都满足硬行程和速度约束。
- 版本核验：Crossref 显示正式期刊出版日期为 2026-07-20；2025 年 opticaopen 版本是预印本。
- 证据：B+。

#### 15. Choi 等，2026，JOSA A

- 论文：[TURBO-RL: turbulence mitigation using reinforcement learning for severe optical aberrations](https://doi.org/10.1364/JOSAA.568108)
- 做法：用 CNN 与 RL 从导星图像估计并校正严重湍流，目标是减少传统 WFS 依赖。
- 报告范围：论文展示了非常强的湍流和低光子数场景。
- 局限：论文较短，底层数据未公开，只能向作者申请；外部复现和硬件证据仍有限。
- 证据：B。

#### 16. Nousiainen 等，2026，Astronomy & Astrophysics

- 论文：[Focal plane wavefront control with model-based reinforcement learning](https://doi.org/10.1051/0004-6361/202558504)
- 做法：PO4NCPA 使用连续的相位多样性焦平面图像，学习静态和动态非共路像差校正。
- 结果边界：静态情况接近理想；动态情况在主要焦平面指标上与“最小二乘重构 + 一步延迟积分器”接近，但高阶波前误差更大。
- 意义：焦平面 RL 已发展到动态非共路像差，不能再用“首次焦平面 RL”作为主张。
- 局限：仍为仿真，且并非所有指标都优于经典方法。
- 证据：B+。

#### 17. Dray 等，2026，SPIE

- 论文：[Deep learning-based reconstructor coupled with reinforcement learning for adaptive optics in LEO optical links](https://doi.org/10.1117/12.3103665)
- 做法：把深度学习重构器与 PO4AO 式预测控制结合，用于低轨光通信中的低光子数和强闪烁场景。
- 意义：监督重构 + RL 的组合正在扩展到光通信，不只存在于天文 AO。
- 局限：会议论文，当前可访问证据以摘要和会议元数据为主，独立复现与完整比较不足。
- 证据：C+。

## 6. 强控制基线：RL 至少要和谁比

只和“不调参的普通积分器”比较，无法证明强化学习有价值。至少需要以下基线。

#### 18. Paschall 与 Anderson，1993

- 论文：[Linear quadratic Gaussian control of a deformable mirror adaptive optics system with time-delayed measurements](https://doi.org/10.1364/AO.32.006347)
- 作用：很早就把测量时延写进 LQG 控制；说明“考虑时延”不是 RL 专属能力。
- 证据：B，奠基理论与数值研究。

#### 19. Kulcsár 等，2006

- 论文：[Optimal control, observers and integrators in adaptive optics](https://doi.org/10.1364/OE.14.007464)
- 作用：建立 Kalman 观测器、最小方差控制和积分器之间的统一框架。
- 项目要求：固定风场、线性近似和已知噪声下，LQG/Kalman 是必须击败的预测控制基线。
- 证据：A−。

#### 20. Konnik 与 De Doná，2015

- 论文：[Feasibility of Constrained Receding Horizon Control Implementation in Adaptive Optics](https://doi.org/10.1109/TCST.2014.2324179)
- 作用：通过带约束的二次规划显式处理执行器行程、变化量和耦合。
- 项目要求：如果 RL 的卖点是“安全和约束”，必须与这种显式约束控制比较，不能只在奖励里加罚分。
- 证据：B+。

#### 21. Haffert 等，2021

- 论文：[Data-driven subspace predictive control of adaptive optics for high-contrast imaging](https://doi.org/10.1117/1.JATIS.7.2.029001)
- 作用：只用闭环波前误差和可变形镜命令在线学习线性预测控制，并在 MagAO-X 实验台验证。
- 项目要求：这是“用历史遥测学习剩余动态”最强的非 RL 对照之一。
- 证据：A。

#### 22. van Kooten 等，2022

- 论文：[Predictive wavefront control on Keck II adaptive optics bench: on-sky coronagraphic results](https://doi.org/10.1117/1.JATIS.8.2.029006)
- 作用：给出 Keck II 的真实观测预测控制结果。
- 项目要求：如果研究声称 RL 利用了冻结流或时间相关性，应与这类成熟预测控制比较。
- 证据：A。

#### 23. Poyneer 等，2023

- 论文：[Laboratory demonstration of the prediction of wind-blown turbulence by adaptive optics at 8 kHz with use of LQG control](https://doi.org/10.1364/AO.474730)
- 作用：在 8 kHz 实验台上比较预测 Fourier-LQG 和积分器，并同时检查遥测和焦平面 Strehl。
- 项目要求：实时延迟和风场预测不能只在慢速玩具环境中验证。
- 证据：A。

## 7. 三项通用 RL 方法依据

这些论文不是 AO 论文，但解释了蓝图里算法名称从哪里来。

#### 24. SAC

- 论文：[Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)
- 核心：利用历史数据反复学习连续动作，并通过熵鼓励探索。
- 边界：原始 SAC 只有普通动作范围，不提供可变形镜硬安全保证。
- 证据：A，机器学习领域的奠基会议论文。

#### 25. 剩余强化学习

- 论文：[Residual Reinforcement Learning for Robot Control](https://doi.org/10.1109/ICRA.2019.8794127)
- 核心：最终命令等于传统控制器命令加上 RL 学到的小修正。
- 边界：这是跨领域已有范式，因此本项目不能把“相加结构”本身当创新。
- 证据：A−，含真实机器人实验。

#### 26. 约束策略优化

- 论文：[Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html)
- 核心：把任务收益和约束代价分开优化。
- 边界：它主要约束期望累计代价，仍不保证每个时刻绝不越过镜面行程；硬限制更适合动作投影、二次规划安全层或紧急回退。
- 证据：A。

## 8. 焦平面与全息观测的覆盖检查

#### 27. Orban de Xivry 等，2021

- 论文：[Focal plane wavefront sensing using machine learning](https://doi.org/10.1093/mnras/stab1634)
- 做法：使用合焦与已知离焦的成对图像，以神经网络回归 Zernike 系数或相位图。
- 项目意义：仅凭一张普通合焦光斑，可能存在相位符号和可辨识性问题；焦平面观测最好包含已知相位多样性或其他物理信息。
- 边界：这是监督学习感知研究，不是强化学习控制。
- 证据：B。

#### 28. Zepp 等，2022

- 论文：[Simulation-based design optimization of the holographic wavefront sensor in closed-loop adaptive optics](https://doi.org/10.37188/lam.2022.027)
- 做法：研究全息波前传感器在闭环 AO 中的设计与优化。
- 项目意义：全息测量进入 AO 闭环并不是空白，但本轮没有在这篇或精确检索结果中发现强化学习控制。
- 边界：它不是离轴数字全息复场 + RL 的直接先例，也不能单独证明该交叉方向无人做过。
- 证据：B。

## 9. 来源核验报告

### 9.1 身份与版本核验

- 26 个带 DOI 的重点来源已通过 Crossref DOI 接口逐条核验，26/26 的 DOI、题名和年份相符。
- 其中 14 个最关键 AO–RL 来源同时通过 OpenAlex 核验，未发现 DOI–题名冲突。
- Semantic Scholar 在本轮多次返回 HTTP 429，因此只有部分来源成功交叉匹配。按照 ARS 规则，这属于数据库暂时降级，不能把未返回结果写成“论文不存在”。
- arXiv 已核验 `2604.00993` 和 `2606.10771` 的题名与作者。两者均已有 A&A 期刊 DOI，报告优先引用期刊版本。
- 2025 年的 Action-Regularized RL 预印本已由 2026 年 JOSA B 正式期刊版本替代。

### 9.2 出版与撤稿风险

- 纳入的期刊和会议均可在出版社、Crossref、PMLR 或正式 arXiv 元数据中核验；本轮未发现掠夺性期刊警报。
- 本轮未发现这些重点来源的撤稿通知或题名版本冲突。
- “未发现撤稿”只表示在本轮检索截止日未见警报，不等于未来永远不会出现更正或撤稿。

### 9.3 利益冲突与数据可得性

- Durech 等 2021 的全文披露一名作者与 Seymour Vision 相关。
- TURBO-RL 的出版社页面说明底层数据当前不公开，可向作者申请，这降低了独立复现便利性。
- 多数其他论文的可访问元数据未显示明显利益冲突警报，但本轮没有逐篇审计全部作者的资金、专利和商业关系，因此不能写成“确定没有利益冲突”。
- PO4AO 2024 论文公开了实现代码和实时控制要求，是本语料中复现信息较完整的工作之一。

## 10. 证据分布偏差提示

`DISTRIBUTIONAL_SKEW_ADVISORY`

- **时间集中**：17 项直接 AO–RL 研究中，15 项发表于 2021–2026 年。这个领域变化很快，旧综述容易漏掉 2024–2026 的关键进展。
- **验证集中**：大多数成果仍以仿真或单一实验台为主；真实天文观测的 RL 期刊证据目前主要来自单一 PO4AO 技术路线。
- **团队集中**：PO4AO、Pou、Parvizi 等少数研究团队贡献了多篇连续论文。论文数量不等于同等数量的独立复现。
- **基线不统一**：不同论文使用单增益积分器、最佳增益积分器、逐模态增益、拥有理想大气知识的 LQG 或不同计算预算，性能倍数不能直接横向比较。
- **目标不统一**：WFS 残差、Strehl、日冕对比度、单模光纤耦合和显微图像质量不是同一个目标。一个指标提高，不能自动代表另一个也提高。

## 11. 对当前蓝图的 Phase 2 约束

以下是文献已经能够确定的事实，不是 Phase 3 的最终创新方案。

### 已经不能单独声称为创新

- 使用 SAC 控制 AO；
- 使用多智能体拆分高维镜面；
- 使用历史观测和历史动作；
- 让 RL 输出增量可变形镜命令；
- 把 RL 修正叠加在积分器上；
- 把监督学习重构器与 RL 预测控制组合；
- 使用焦平面图像或图像质量作为反馈；
- 在奖励中加入动作大小或平滑惩罚；
- 从仿真走向实验台或单次真实天文观测。

### 仍值得 Phase 3 逐项核查的候选空缺

1. **全息复场观测的独特价值**：离轴数字全息同时提供振幅和相位信息后，RL 是否仍比明确的相位投影、LQG、MPC 或数据驱动预测控制更有价值？
2. **真正公平的残差控制比较**：在相同环境交互次数、总计算量和部署时延下，残差 RL 是否还能超过逐模态积分器、LQG、MPC 和子空间预测控制？
3. **可证伪的时间价值**：打乱历史帧或把折扣因子设为零后，性能是否明显下降？如果不下降，所谓 RL 优势可能只是逐帧非线性拟合。
4. **真实可部署的安全闭环**：同时记录实际执行命令、饱和、丢帧、时延抖动和回退事件，并使用硬动作投影或安全层，而不是只在奖励中软惩罚。
5. **跨仿真—真实的受控验证**：不是直接宣称零样本迁移，而是区分离线训练、影子模式、低阶受限控制、实验台闭环和最终真实系统复现。

这些只能叫“候选空缺”。Phase 3 需要把每个候选与最相近论文逐格比较后，才能决定最终研究问题。

## 12. Phase 3 的输入清单

Phase 2 已为下一阶段准备好四类输入：

1. 17 项直接 AO–RL 近邻研究；
2. 6 项必须认真实现的强控制基线；
3. 观测、动作、奖励、时延和验证层级的对照信息；
4. 一个尚未找到直接同题论文、但仍需谨慎确认的“全息复场观测 + RL 闭环”交叉候选。

Phase 3 应完成：

- 建立“本项目 vs. 最近邻论文”的创新矩阵；
- 删除已经被文献覆盖的宽泛创新表述；
- 在 SAC、PPO、PO4AO 式模型强化学习和非 RL 预测控制之间做算法选择；
- 把最终研究问题收缩为一个可以用最小实验否证的问题；
- 决定先做高速 WFS/全息主环，还是焦平面低速外环。

## 13. 本阶段限制与 AI 使用说明

- 本报告由 AI 辅助完成检索、去重、DOI 核验和中文整理。
- 重要题名和 DOI 已通过权威元数据或出版社来源核验，但并非每篇论文都完成了逐公式、逐图表的人工全文复核。
- 搜索受数据库覆盖、索引延迟、访问权限和关键词影响。“没有找到”不等于“绝对不存在”。
- 任何“首次”“唯一”“显著优于所有方法”的论文表述，在正式投稿前都应由研究者再次阅读原文、检查最新引文和复现实验。
