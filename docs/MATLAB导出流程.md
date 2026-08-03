# MATLAB 导出流程

本项目目前有两类 MATLAB 导出：静态仿真数据和真实离轴全息复光场。两者都可以继续用于软件检查，但都不是动态 RL 轨迹。

## 状态说明

> `matlab/simulate_gaussian_turbulence_dataset.m` 的 S0 物理单元测试已于 2026-08-03 通过（15/15）。当前输出可用于物理接口验证和静态基线，但仍没有动作后的下一时刻观测，不能冒充动态 RL 数据或真实硬件结论。

已经完成的 S0 修复：

1. 每段先在空间域施加相位屏，再做 Fresnel 传播；
2. 第一块以及后续所有相位屏都实际参与传播；
3. `r0` 改为按完整传播距离计算；
4. 无湍流参考场传播到与湍流场相同的接收面；
5. 增加焦平面 Strehl 和桶内功率指标，已验证已知散焦的反相位补偿会改善指标；
6. 增加 SLM 相位范围、单步变化、量化和帧延迟模型；
7. 增加按完整场景或回合分组的划分函数，防止相邻帧泄漏。

详细验收结果见 [S0 物理门验证记录](S0物理门验证记录.md)。配置文件保持 `allow_scientific_claims: false`，因为 S1 动态环境和真实硬件验证尚未完成。

## 1. 静态仿真数据：用于 S0 验证与静态基线

脚本：

```text
matlab/simulate_gaussian_turbulence_dataset.m
```

当前用途：

```text
检查传播物理、MATLAB → HDF5 → PyTorch 接口，以及静态监督基线
```

建议只生成很小的数据：

```matlab
cd('C:\Users\Lintianze\OneDrive\Desktop\AO_DL_Turbulence')
addpath(fullfile(pwd, 'matlab'))

simulate_gaussian_turbulence_dataset( ...
    fullfile(pwd, 'data', 'raw_matlab_exports', 'sim_gaussian_v1.h5'), ...
    10, ...
    64)
```

参数含义：

```text
第 1 个参数：输出 HDF5 文件路径
第 2 个参数：生成帧数
第 3 个参数：图像尺寸 N，输出为 N × N
```

该导出器仍然逐帧生成相互独立的静态场。即使扩大到 1000 帧，也不会自动变成 RL 回合；正式动态数据要等 S1 环境实现后再生成。

## 2. 未来的动态纯仿真数据

正式 RL 数据不能只是互不相关的图片。每个仿真回合必须保存：

```text
当前观测 → 控制器动作 → 传播与执行器响应 → 下一时刻观测
```

动态仿真至少应包含：

- 连续演化的湍流相位屏；
- SLM 模态动作或相位动作；
- 0—3 帧可配置延迟；
- 相机噪声、SLM 量化、饱和和稳定时间；
- 动作后的相位残差与焦平面指标；
- 完整回合编号、随机种子和时间步。

动态仿真脚本和命令目前尚未实现。本文件不预先写一个不存在的入口；实现完成后，再补充确切命令。数据字段见[数据格式说明](数据格式.md)。

## 3. 导出真实离轴全息复光场

脚本：

```text
matlab/export_real_offaxis_validation_dataset.m
```

用途：

```text
读取 E:\加扩束镜\2倍放大 下的真实 PNG 离轴全息图，
使用 offaxisholo 同源的频域裁剪、补零和逆傅里叶变换流程，
导出真实复光场强度和相位。
```

运行示例：

```matlab
cd('C:\Users\Lintianze\OneDrive\Desktop\AO_DL_Turbulence')
addpath(fullfile(pwd, 'matlab'))

export_real_offaxis_validation_dataset( ...
    fullfile(pwd, 'data', 'real_validation', 'real_offaxis_2x_validation.h5'), ...
    'E:\加扩束镜\2倍放大', ...
    200, ...
    [256 256])
```

参数含义：

```text
第 1 个参数：输出 HDF5 文件路径
第 2 个参数：真实实验 PNG 根目录
第 3 个参数：每个温差最多导出多少帧
第 4 个参数：导出的复光场尺寸 [height width]
```

当前默认处理 11 个温差文件夹：

```text
温差40, 温差80, 温差100, 温差140, 温差180,
温差200, 温差240, 温差280, 温差310, 温差330, 温差375
```

每组导出 200 帧时总计 2200 帧。

### 结果边界

这些真实数据没有同步记录 SLM 动作和动作后的下一帧，因此：

- 可以做真实域分布检查和离轴重建验证；
- 可以作为冻结模型的离线影子输入；
- 不能直接训练闭环 RL；
- 不能证明真实补偿效果已经提高。

## 4. 真实复光场重建方法

真实导出脚本复用 `offaxisholo` 的核心处理思想：

```text
1. 读取 PNG 灰度图
2. 对全息图做 fftshift(fft2(fftshift(hologram)))
3. 在频域裁剪一阶项
4. 把裁剪结果补零到原图大小并放到频谱中心
5. ifftshift(ifft2(ifftshift(paddedSpectrum))) 得到复光场
6. 导出 abs(field)^2 作为强度
7. 导出 angle(field) 作为相位
```

默认参考裁剪窗口：

```text
[179 72 236 120]
```

默认自动裁剪参数：

```text
CropMode = auto
AutoSearchRadius = 40
AutoDetectSampleCount = 5
```

裁剪窗口和自动检测结果属于数据来源的一部分，正式数据必须与每一帧或每个实验块一起保存。
