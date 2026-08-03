function nextPhase = ao_taylor_frozen_flow_step(phase, samplePitch, shiftX, shiftY, rho, innovation)
%AO_TAYLOR_FROZEN_FLOW_STEP Advance one periodic Taylor frozen-flow frame.
%
% shiftX and shiftY are physical displacements in metres. rho=1 gives the
% strict frozen-flow hypothesis. rho<1 mixes in an independent innovation.

if nargin < 5 || isempty(rho)
    rho = 1;
end
if nargin < 6
    innovation = [];
end
validateattributes(phase, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'}, ...
    mfilename, 'phase');
validateattributes(samplePitch, {'numeric'}, {'scalar', 'real', 'positive', 'finite'}, ...
    mfilename, 'samplePitch');
validateattributes(shiftX, {'numeric'}, {'scalar', 'real', 'finite'}, mfilename, 'shiftX');
validateattributes(shiftY, {'numeric'}, {'scalar', 'real', 'finite'}, mfilename, 'shiftY');
validateattributes(rho, {'numeric'}, {'scalar', 'real', '>=', 0, '<=', 1}, mfilename, 'rho');
[height, width] = size(phase);
if height ~= width || mod(height, 2) ~= 0
    error('ao:InvalidGrid', 'phase must be a square array with an even side length.');
end
if rho < 1 && isempty(innovation)
    error('ao:MissingInnovation', 'innovation is required when rho is less than one.');
end
if ~isempty(innovation) && ~isequal(size(innovation), size(phase))
    error('ao:InvalidInnovation', 'innovation must have the same size as phase.');
end

frequency = ifftshift((-height/2:height/2-1) / (height * samplePitch));
[FX, FY] = meshgrid(frequency, frequency);
phaseRamp = exp(-1i * 2 * pi .* (FX .* shiftX + FY .* shiftY));
translated = real(ifft2(fft2(phase) .* phaseRamp));
if rho == 1
    nextPhase = translated;
else
    nextPhase = rho .* translated + sqrt(1 - rho^2) .* innovation;
end
nextPhase = nextPhase - mean(nextPhase, 'all');
end
