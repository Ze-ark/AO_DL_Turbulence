# MATLAB 导出流程

本项目有两类 MATLAB 数据导出脚本：仿真数据导出和真实实验复光场导出。

## 1. 导出仿真训练数据

脚本：

```text
matlab/simulate_gaussian_turbulence_dataset.m
```

用途：

```text
生成无湍流高斯光目标和对应湍流复光场
```

运行示例：

```matlab
cd('C:\Users\Lintianze\OneDrive\Desktop\AO_DL_Turbulence')
addpath(fullfile(pwd, 'matlab'))

simulate_gaussian_turbulence_dataset( ...
    fullfile(pwd, 'data', 'raw_matlab_exports', 'sim_gaussian_v1.h5'), ...
    1000, ...
    256)
```

参数含义：

```text
第 1 个参数：输出 HDF5 文件路径
第 2 个参数：生成帧数
第 3 个参数：图像尺寸 N，输出为 N x N
```

快速测试可用：

```matlab
simulate_gaussian_turbulence_dataset( ...
    fullfile(pwd, 'data', 'raw_matlab_exports', 'sim_gaussian_test.h5'), ...
    10, ...
    64)
```

## 2. 导出真实实验验证数据

脚本：

```text
matlab/export_real_offaxis_validation_dataset.m
```

用途：

```text
读取 E:\加扩束镜\2倍放大 下的真实 PNG 离轴全息图，
使用 offaxisholo 同源的频域裁剪、补零、ifft 流程，
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

当前默认会处理这些温差文件夹：

```text
温差40, 温差80, 温差100, 温差140, 温差180,
温差200, 温差240, 温差280, 温差310, 温差330, 温差375
```

如果每个温差导出 200 帧，总帧数为：

```text
11 * 200 = 2200
```

## 真实复光场重建方法

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

默认使用自动裁剪：

```text
CropMode = auto
AutoSearchRadius = 40
AutoDetectSampleCount = 5
```

