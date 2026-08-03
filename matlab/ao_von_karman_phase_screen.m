function phase = ao_von_karman_phase_screen(N, Cn2, segmentLength, params, X, Y)
%AO_VON_KARMAN_PHASE_SCREEN Generate one modified Von Karman phase screen.
%
% The high-frequency FFT screen is supplemented with low-frequency
% subharmonics. The screen is spatial-domain phase in radians.

validateattributes(N, {'numeric'}, {'scalar', 'integer', 'even', '>=', 4}, ...
    mfilename, 'N');
validateattributes(Cn2, {'numeric'}, {'scalar', 'real', 'nonnegative', 'finite'}, ...
    mfilename, 'Cn2');
validateattributes(segmentLength, {'numeric'}, {'scalar', 'real', 'positive', 'finite'}, ...
    mfilename, 'segmentLength');
if ~isequal(size(X), [N, N]) || ~isequal(size(Y), [N, N])
    error('ao:InvalidGrid', 'X and Y must both be N-by-N arrays.');
end

if Cn2 == 0
    phase = zeros(N, N);
    return
end

r0Segment = ao_fried_parameter(Cn2, params.lambda, segmentLength);
deltaF = 1 / params.screenSize;
frequency = (-N/2:N/2-1) * deltaF;
[FX, FY] = meshgrid(frequency, frequency);
powerSpectrum = phase_power_spectrum(hypot(FX, FY), r0Segment, params);
powerSpectrum(N/2+1, N/2+1) = 0;

coefficients = (randn(N) + 1i * randn(N)) .* sqrt(powerSpectrum) * deltaF;
highFrequency = real(fftshift(ifft2(ifftshift(coefficients)))) * N^2;
lowFrequency = subharmonic_component(r0Segment, params, X, Y);
phase = highFrequency + lowFrequency;
phase = phase - mean(phase, 'all');
end

function spectrum = phase_power_spectrum(radialFrequency, r0, params)
innerFrequency = 5.92 / (2 * pi * params.l0);
outerFrequency = 1 / params.L0;
spectrum = 0.023 * r0^(-5/3) ...
    .* exp(-(radialFrequency / innerFrequency).^2) ...
    ./ (radialFrequency.^2 + outerFrequency^2).^(11/6);
end

function phase = subharmonic_component(r0, params, X, Y)
phase = zeros(size(X));
for level = 1:params.subharmonicLevels
    deltaF = 1 / (3^level * params.screenSize);
    frequency = (-1:1) * deltaF;
    [FX, FY] = meshgrid(frequency, frequency);
    powerSpectrum = phase_power_spectrum(hypot(FX, FY), r0, params);
    powerSpectrum(2, 2) = 0;
    coefficients = (randn(3) + 1i * randn(3)) .* sqrt(powerSpectrum) * deltaF;
    component = zeros(size(X));
    for index = 1:9
        component = component + coefficients(index) ...
            .* exp(1i * 2 * pi .* (FX(index) .* X + FY(index) .* Y));
    end
    phase = phase + real(component);
end
phase = phase - mean(phase, 'all');
end
