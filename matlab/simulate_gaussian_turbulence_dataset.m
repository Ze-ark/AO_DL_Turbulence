function simulate_gaussian_turbulence_dataset(outputPath, numFrames, N)
%SIMULATE_GAUSSIAN_TURBULENCE_DATASET Export Gaussian-beam turbulence data.
%
% This script follows the structure of dwsdr's Gaussian beam atmospheric
% turbulence example: corrected Von Karman phase screens, subharmonic
% low-frequency compensation, and split-step frequency-domain propagation.
% It rewrites the workflow as a batch HDF5 exporter for PyTorch training.
%
% Example:
%   simulate_gaussian_turbulence_dataset( ...
%       'data/raw_matlab_exports/sim_gaussian_v1.h5', 1000, 256)

if nargin < 1 || isempty(outputPath)
    outputPath = fullfile('data', 'raw_matlab_exports', 'sim_gaussian_v1.h5');
end
if nargin < 2 || isempty(numFrames)
    numFrames = 1000;
end
if nargin < 3 || isempty(N)
    N = 256;
end

rng(42);
outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if exist(outputPath, 'file')
    delete(outputPath);
end

params.z = 3300;                  % propagation distance, m
params.w0 = 0.012;                % beam waist radius, m
params.Cn2Base = 5e-15;           % refractive-index structure constant
params.lambda = 1550e-9;          % wavelength, m
params.numScreens = 8;
params.L0 = 10;                   % outer scale, m
params.l0 = 0.01;                 % inner scale, m
params.screenSize = 1.0;          % transverse grid size, m
params.subharmonicLevels = 3;

delta = params.screenSize / N;
x = (-N/2:N/2-1) * delta;
[X, Y] = meshgrid(x, x);
r = sqrt(X.^2 + Y.^2);
k = 2 * pi / params.lambda;
Eclean = exp(-(r.^2) ./ params.w0^2);
Iclean = single(abs(Eclean).^2);
Pclean = single(angle(Eclean));

shape = [N, N, numFrames];
h5create(outputPath, '/input/intensity_turb', shape, 'Datatype', 'single', 'ChunkSize', [N, N, 1]);
h5create(outputPath, '/input/phase_turb', shape, 'Datatype', 'single', 'ChunkSize', [N, N, 1]);
h5create(outputPath, '/target/intensity_clean', shape, 'Datatype', 'single', 'ChunkSize', [N, N, 1]);
h5create(outputPath, '/target/phase_clean', shape, 'Datatype', 'single', 'ChunkSize', [N, N, 1]);
h5create(outputPath, '/meta/frame_id', [numFrames, 1], 'Datatype', 'int64');
h5create(outputPath, '/meta/turbulence_strength', [numFrames, 1], 'Datatype', 'single');
h5create(outputPath, '/meta/r0_or_equivalent', [numFrames, 1], 'Datatype', 'single');

frameIds = int64((0:numFrames-1)');
strengths = single(linspace(0.2, 1.0, numFrames)');
r0Values = zeros(numFrames, 1, 'single');

for frame = 1:numFrames
    strength = double(strengths(frame));
    Cn2 = params.Cn2Base * strength;
    Eturb = propagate_with_turbulence(Eclean, Cn2, params, X, Y);
    r0Values(frame) = single((0.4229 * k^2 * Cn2 * (params.z / params.numScreens))^(-3/5));

    h5write(outputPath, '/input/intensity_turb', single(abs(Eturb).^2), [1, 1, frame], [N, N, 1]);
    h5write(outputPath, '/input/phase_turb', single(angle(Eturb)), [1, 1, frame], [N, N, 1]);
    h5write(outputPath, '/target/intensity_clean', Iclean, [1, 1, frame], [N, N, 1]);
    h5write(outputPath, '/target/phase_clean', Pclean, [1, 1, frame], [N, N, 1]);
end

h5write(outputPath, '/meta/frame_id', frameIds);
h5write(outputPath, '/meta/turbulence_strength', strengths);
h5write(outputPath, '/meta/r0_or_equivalent', r0Values);
h5writeatt(outputPath, '/', 'description', 'Simulated Gaussian beam turbulence compensation dataset');
h5writeatt(outputPath, '/', 'layout', 'Arrays are stored as [height, width, frame] for MATLAB; Python loader transposes if needed.');
fprintf('Wrote %d frames to %s\n', numFrames, outputPath);
end

function U = propagate_with_turbulence(E0, Cn2, params, X, Y)
N = size(E0, 1);
k = 2 * pi / params.lambda;
deltz = params.z / params.numScreens;
delta = params.screenSize / N;
del_f = 1 / (N * delta);
fx = (-N/2:N/2-1) * del_f;
[kx, ky] = meshgrid(2*pi*fx, 2*pi*fx);
[~, ka] = cart2pol(kx, ky);
km = 5.92 / params.l0;
k0 = 2 * pi / params.L0;
PSD_phi = 0.033 * Cn2 * exp(-(ka / km).^2) ./ (ka.^2 + k0.^2).^(11/6);
PSD_phi(N/2+1, N/2+1) = 0;
cn = 2 * pi * k^2 * deltz * PSD_phi * (2 * pi * del_f)^2;

freq = linspace(-params.screenSize/2, params.screenSize/2, N) * N;
[kethi, nenta] = meshgrid(freq, freq);
H = exp(1i * k * deltz .* (1 - (params.lambda^2) .* (kethi.^2 + nenta.^2) / 2));
G = fftshift(fft2(E0));

for screen = 1:params.numScreens
    phz_hi = real(ift2_ao((randn(N) + 1i * randn(N)) .* sqrt(cn), 1));
    phz_lo = subharmonic_phase(Cn2, params, X, Y, k, deltz);
    phz = phz_hi + phz_lo;
    if screen > 1
        G = G .* exp(1i * phz);
    end
    G = G .* H;
end
U = ifft2(G);
end

function phz_lo = subharmonic_phase(Cn2, params, X, Y, k, deltz)
N = size(X, 1);
phz_lo = zeros(N, N);
for p = 1:params.subharmonicLevels
    del_fp = 1 / (3^p * params.screenSize);
    fx1 = (-1:1) * del_fp;
    [kx1, ky1] = meshgrid(2*pi*fx1, 2*pi*fx1);
    [~, k1] = cart2pol(kx1, ky1);
    km = 5.92 / params.l0;
    k0 = 2 * pi / params.L0;
    PSD_phi1 = 0.033 * Cn2 * exp(-(k1 / km).^2) ./ (k1.^2 + k0.^2).^(11/6);
    PSD_phi1(2, 2) = 0;
    cn1 = 2 * pi * k^2 * deltz .* PSD_phi1 .* (2*pi*del_fp)^2;
    cn1 = (randn(3) + 1i * randn(3)) .* sqrt(cn1);
    SH = zeros(N, N);
    for ii = 1:9
        SH = SH + cn1(ii) * exp(1i * (kx1(ii) * X + ky1(ii) * Y));
    end
    phz_lo = phz_lo + SH;
end
phz_lo = real(phz_lo) - mean(real(phz_lo(:)));
end
