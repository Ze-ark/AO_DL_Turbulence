function result = demo_three_turbulence_animation(outputDir, numFrames)
%DEMO_THREE_TURBULENCE_ANIMATION 生成三类动态湍流的同步对比动画。
%
% 动画用于解释三种未补偿仿真条件的时间差异，不是控制器或强化学习结果。

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile('outputs', 'three_turbulence_regimes_demo');
end
if nargin < 2 || isempty(numFrames)
    numFrames = 81;
end
validateattributes(numFrames, {'numeric'}, ...
    {'scalar', 'integer', '>=', 2}, mfilename, 'numFrames');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

gifPath = fullfile(outputDir, '三种动态湍流同步对比动画.gif');
if exist(gifPath, 'file')
    delete(gifPath);
end

previousRng = rng;
rngCleanup = onCleanup(@() rng(previousRng));
rng(20260918, 'twister');

gridSize = 128;
frameIntervalS = 0.02;
params = ao_default_simulation_params();
params.screenSize = 0.4;
params.L0 = 10;
params.l0 = 0.01;
params.subharmonicLevels = 0;
samplePitchM = params.screenSize / gridSize;
coordinateM = (-gridSize/2:gridSize/2-1) .* samplePitchM;
[X, Y] = meshgrid(coordinateM, coordinateM);

desiredR0M = 0.12;
referenceLengthM = 100;
wavenumber = 2 * pi / params.lambda;
Cn2 = desiredR0M^(-5/3) / (0.423 * wavenumber^2 * referenceLengthM);
initialPhase = ao_von_karman_phase_screen( ...
    gridSize, Cn2, referenceLengthM, params, X, Y);
phase = repmat(initialPhase, 1, 1, 3);

titles = { ...
    '① 冻结流：纹理形状不变，只整体平移', ...
    '② 沸腾型：平移时纹理不断变形', ...
    '③ 变风型：纹理变形，风向箭头也转动'};
baseSpeedMps = [0.45, 0.55, 0.60];
baseDirectionDeg = [15, 60, 75];
rho = [1.0, 0.995, 0.995];
speedFraction = [0, 0, 0.15];
directionAmplitudeDeg = [0, 0, 15];
periodFrames = [0, 0, 80];
phaseDeg = [0, 0, 135];

colorLimit = max(abs(initialPhase), [], 'all') * 1.5;
animationFigure = figure(Visible='off', Color='white', ...
    Position=[100, 100, 1500, 520]);
figureCleanup = onCleanup(@() close(animationFigure));
layout = tiledlayout(animationFigure, 1, 3, ...
    TileSpacing='compact', Padding='compact');
title(layout, '三种动态大气湍流同步演化（未补偿）', ...
    FontWeight='bold', FontSize=18);

axisHandles = gobjects(1, 3);
imageHandles = gobjects(1, 3);
arrowHandles = gobjects(1, 3);
for regimeIndex = 1:3
    axisHandles(regimeIndex) = nexttile(layout);
    imageHandles(regimeIndex) = imagesc(axisHandles(regimeIndex), ...
        coordinateM, coordinateM, phase(:, :, regimeIndex));
    axis(axisHandles(regimeIndex), 'image');
    set(axisHandles(regimeIndex), 'YDir', 'normal');
    clim(axisHandles(regimeIndex), [-colorLimit, colorLimit]);
    colormap(axisHandles(regimeIndex), turbo(256));
    xlabel(axisHandles(regimeIndex), 'x / m');
    ylabel(axisHandles(regimeIndex), 'y / m');
    hold(axisHandles(regimeIndex), 'on');
    arrowHandles(regimeIndex) = quiver(axisHandles(regimeIndex), ...
        -0.15, -0.15, 0.06, 0, 0, Color='white', ...
        LineWidth=3, MaxHeadSize=1.5);
end
colorbarHandle = colorbar(axisHandles(3));
colorbarHandle.Layout.Tile = 'east';
colorbarHandle.Label.String = '相位 / rad';

temporaryPath = fullfile(outputDir, '_animation_frame.png');
for frameIndex = 1:numFrames
    timeS = (frameIndex - 1) * frameIntervalS;
    for regimeIndex = 1:3
        [speed, direction] = wind_at_frame( ...
            baseSpeedMps(regimeIndex), baseDirectionDeg(regimeIndex), ...
            frameIndex - 1, speedFraction(regimeIndex), ...
            directionAmplitudeDeg(regimeIndex), periodFrames(regimeIndex), ...
            phaseDeg(regimeIndex));
        imageHandles(regimeIndex).CData = phase(:, :, regimeIndex);
        arrowLength = 0.07 * speed / max(baseSpeedMps);
        arrowHandles(regimeIndex).UData = arrowLength * cosd(direction);
        arrowHandles(regimeIndex).VData = arrowLength * sind(direction);
        title(axisHandles(regimeIndex), {titles{regimeIndex}, ...
            sprintf('t = %.2f s，风速 %.2f m/s，风向 %.1f°', ...
            timeS, speed, direction)}, FontWeight='bold');
    end
    drawnow;
    exportgraphics(animationFigure, temporaryPath, Resolution=120);
    frameRgb = imread(temporaryPath);
    [indexedFrame, frameMap] = rgb2ind(frameRgb, 256);
    if frameIndex == 1
        imwrite(indexedFrame, frameMap, gifPath, 'gif', ...
            LoopCount=Inf, DelayTime=0.10);
    else
        imwrite(indexedFrame, frameMap, gifPath, 'gif', ...
            WriteMode='append', DelayTime=0.10);
    end

    if frameIndex == numFrames
        continue
    end
    for regimeIndex = 1:3
        [speed, direction] = wind_at_frame( ...
            baseSpeedMps(regimeIndex), baseDirectionDeg(regimeIndex), ...
            frameIndex - 1, speedFraction(regimeIndex), ...
            directionAmplitudeDeg(regimeIndex), periodFrames(regimeIndex), ...
            phaseDeg(regimeIndex));
        shiftX = speed * frameIntervalS * cosd(direction);
        shiftY = speed * frameIntervalS * sind(direction);
        if rho(regimeIndex) < 1
            innovation = ao_von_karman_phase_screen( ...
                gridSize, Cn2, referenceLengthM, params, X, Y);
        else
            innovation = [];
        end
        phase(:, :, regimeIndex) = ao_taylor_frozen_flow_step( ...
            phase(:, :, regimeIndex), samplePitchM, shiftX, shiftY, ...
            rho(regimeIndex), innovation);
    end
end
if exist(temporaryPath, 'file')
    delete(temporaryPath);
end

result.gifPath = gifPath;
result.numFrames = numFrames;
result.simulatedDurationS = (numFrames - 1) * frameIntervalS;
result.note = '未补偿的仿真相位演示，不是控制器或RL性能比较结果';
fprintf('动态对比图：%s\n', gifPath);
clear figureCleanup rngCleanup
end

function [speedMps, directionDeg] = wind_at_frame( ...
        baseSpeedMps, baseDirectionDeg, frameIndex, ...
        speedFraction, directionAmplitudeDeg, periodFrames, phaseDeg)
if speedFraction == 0 && directionAmplitudeDeg == 0
    speedMps = baseSpeedMps;
    directionDeg = baseDirectionDeg;
    return
end
angleRad = 2 * pi * frameIndex / periodFrames + deg2rad(phaseDeg);
speedMps = baseSpeedMps * (1 + speedFraction * sin(angleRad));
directionDeg = baseDirectionDeg + directionAmplitudeDeg * cos(angleRad);
end
