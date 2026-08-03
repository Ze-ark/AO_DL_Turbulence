function result = demo_s1_dynamic_turbulence(outputDir, numFrames)
%DEMO_S1_DYNAMIC_TURBULENCE Create a visible Taylor frozen-flow demonstration.
%
% The animation shows wrapped pupil phase on the left and the corresponding
% uncorrected focal spot on the right. This is a deterministic demonstration,
% not a controller comparison or an RL result.

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile('outputs', 's1_matlab_demo');
end
if nargin < 2 || isempty(numFrames)
    numFrames = 48;
end
validateattributes(numFrames, {'numeric'}, {'scalar', 'integer', '>=', 2}, ...
    mfilename, 'numFrames');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

gifPath = fullfile(outputDir, 's1_dynamic_turbulence.gif');
summaryPath = fullfile(outputDir, 's1_dynamic_turbulence_summary.png');
metricsPath = fullfile(outputDir, 's1_dynamic_turbulence_metrics.csv');
if exist(gifPath, 'file')
    delete(gifPath);
end

N = 128;
params = ao_default_simulation_params();
params.screenSize = 0.4;
params.L0 = 10;
params.l0 = 0.01;
params.subharmonicLevels = 0;
desiredR0 = 0.12;
referenceLength = 100;
k = 2 * pi / params.lambda;
Cn2 = desiredR0^(-5/3) / (0.423 * k^2 * referenceLength);
samplePitch = params.screenSize / N;
coordinate = (-N/2:N/2-1) * samplePitch;
[X, Y] = meshgrid(coordinate, coordinate);
pupil = hypot(X, Y) <= 0.4 * params.screenSize;

windSpeed = 0.5;
windDirection = 30;
frameInterval = 0.01;
shiftX = windSpeed * frameInterval * cosd(windDirection);
shiftY = windSpeed * frameInterval * sind(windDirection);

previousRng = rng;
rngCleanup = onCleanup(@() rng(previousRng));
rng(42, 'twister');
phase = ao_von_karman_phase_screen( ...
    N, Cn2, referenceLength, params, X, Y);

time = (0:numFrames-1)' * frameInterval;
strehl = zeros(numFrames, 1);
powerInBucket = zeros(numFrames, 1);
phaseRms = zeros(numFrames, 1);
phaseMap = turbo(256);
focusMap = hot(256);

viewer = viewer2d();
viewerCleanup = onCleanup(@() delete(viewer));
imageObject = imageshow([], Parent=viewer, PyramidSmoothing="nearest");

for frameIndex = 1:numFrames
    residualPhase = phase .* pupil;
    [metrics, focalIntensity] = ao_focal_plane_metrics( ...
        pupil .* exp(1i .* residualPhase), pupil, 2);
    strehl(frameIndex) = metrics.strehl;
    powerInBucket(frameIndex) = metrics.powerInBucket;
    phaseRms(frameIndex) = sqrt(mean(residualPhase(pupil).^2));

    wrappedPhase = angle(exp(1i .* residualPhase));
    phaseRgb = indexed_rgb(wrappedPhase, -pi, pi, phaseMap);
    normalizedFocus = focalIntensity ./ max(focalIntensity, [], 'all');
    logFocus = log10(normalizedFocus + 1e-6);
    focusRgb = indexed_rgb(logFocus, -6, 0, focusMap);
    composite = imtile({phaseRgb, focusRgb}, ...
        GridSize=[1, 2], BorderSize=8, BackgroundColor="white");

    imageObject.Data = composite;
    drawnow;
    frameImage = im2uint8(composite);
    [indexedFrame, frameColorMap] = rgb2ind(frameImage, 256);
    if frameIndex == 1
        imwrite(indexedFrame, frameColorMap, gifPath, 'gif', ...
            LoopCount=Inf, DelayTime=0.12);
    else
        imwrite(indexedFrame, frameColorMap, gifPath, 'gif', ...
            WriteMode='append', DelayTime=0.12);
    end

    phase = ao_taylor_frozen_flow_step( ...
        phase, samplePitch, shiftX, shiftY, 1);
end

metricsTable = table(time, strehl, powerInBucket, phaseRms, ...
    VariableNames={'time_s', 'strehl', 'power_in_bucket', 'phase_rms_rad'});
writetable(metricsTable, metricsPath);

summaryFigure = figure(Visible='off', Color='white', Position=[100, 100, 1000, 700]);
figureCleanup = onCleanup(@() close(summaryFigure));
layout = tiledlayout(summaryFigure, 2, 2, TileSpacing='compact', Padding='compact');
title(layout, 'MATLAB S1 泰勒冻结流动态湍流演示');

nexttile(layout);
imagesc(coordinate, coordinate, wrappedPhase);
axis image;
colorbar;
clim([-pi, pi]);
xlabel('x / m');
ylabel('y / m');
title('最后一帧包裹相位 / rad');

nexttile(layout);
imagesc(logFocus);
axis image off;
colorbar;
clim([-6, 0]);
title('最后一帧未补偿焦斑 / log_{10}');

nexttile(layout, [1, 2]);
plot(time, strehl, LineWidth=1.8, DisplayName='Strehl');
hold on;
plot(time, powerInBucket, LineWidth=1.8, DisplayName='桶内功率');
grid on;
xlabel('时间 / s');
ylabel('归一化指标');
ylim([0, 1]);
legend(Location='best');
title(sprintf('风速 %.1f m/s，风向 %d°，r_0 = %.2f m，无补偿', ...
    windSpeed, windDirection, desiredR0));
exportgraphics(summaryFigure, summaryPath, Resolution=160);

result.gifPath = gifPath;
result.summaryPath = summaryPath;
result.metricsPath = metricsPath;
result.numFrames = numFrames;
result.initialStrehl = strehl(1);
result.finalStrehl = strehl(end);
result.minimumStrehl = min(strehl);
result.maximumStrehl = max(strehl);
fprintf('GIF: %s\n', gifPath);
fprintf('Summary: %s\n', summaryPath);
fprintf('Strehl range: %.4f to %.4f\n', result.minimumStrehl, result.maximumStrehl);
end

function rgb = indexed_rgb(values, lowerLimit, upperLimit, colorMap)
normalized = (values - lowerLimit) ./ (upperLimit - lowerLimit);
normalized = min(max(normalized, 0), 1);
indices = floor(normalized * (size(colorMap, 1) - 1)) + 1;
rgb = reshape(colorMap(indices, :), [size(values), 3]);
end
