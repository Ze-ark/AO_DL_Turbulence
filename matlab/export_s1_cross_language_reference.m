function export_s1_cross_language_reference(outputPath)
%EXPORT_S1_CROSS_LANGUAGE_REFERENCE Write a deterministic MATLAB S1 step.

if nargin < 1 || isempty(outputPath)
    outputPath = fullfile('data', 'processed', 's1_matlab_reference.h5');
end
outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if exist(outputPath, 'file')
    delete(outputPath);
end

N = 32;
samplePitch = 0.01;
coordinate = (0:N-1) * samplePitch;
[X, Y] = meshgrid(coordinate, coordinate);
screenWidth = N * samplePitch;
phase = sin(2 * pi .* X ./ screenWidth) ...
    + 0.35 .* cos(4 * pi .* Y ./ screenWidth) ...
    + 0.2 .* sin(2 * pi .* (X + 2 .* Y) ./ screenWidth);
innovation = 0.6 .* cos(6 * pi .* X ./ screenWidth) ...
    + 0.1 .* sin(4 * pi .* (2 .* X - Y) ./ screenWidth);
shiftX = 0.37 * samplePitch;
shiftY = -0.23 * samplePitch;
rho = 0.93;
nextPhase = ao_taylor_frozen_flow_step( ...
    phase, samplePitch, shiftX, shiftY, rho, innovation);

h5create(outputPath, '/phase_initial', [N, N], 'Datatype', 'double');
h5create(outputPath, '/innovation', [N, N], 'Datatype', 'double');
h5create(outputPath, '/phase_next', [N, N], 'Datatype', 'double');
h5write(outputPath, '/phase_initial', phase);
h5write(outputPath, '/innovation', innovation);
h5write(outputPath, '/phase_next', nextPhase);
h5writeatt(outputPath, '/', 'sample_pitch_m', samplePitch);
h5writeatt(outputPath, '/', 'shift_x_m', shiftX);
h5writeatt(outputPath, '/', 'shift_y_m', shiftY);
h5writeatt(outputPath, '/', 'rho', rho);
h5writeatt(outputPath, '/', 'producer', 'MATLAB ao_taylor_frozen_flow_step');
h5writeatt(outputPath, '/', 'python_read_transform', 'transpose_2d_arrays');
fprintf('Wrote S1 MATLAB reference to %s\n', outputPath);
end
