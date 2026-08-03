function [outputField, diagnostics] = ao_split_step_propagate(inputField, Cn2, params, seed)
%AO_SPLIT_STEP_PROPAGATE Alternate spatial phase screens and free propagation.

validateattributes(inputField, {'numeric'}, {'2d', 'nonempty', 'finite'}, ...
    mfilename, 'inputField');
validateattributes(Cn2, {'numeric'}, {'scalar', 'real', 'nonnegative', 'finite'}, ...
    mfilename, 'Cn2');
validateattributes(params.numScreens, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
    mfilename, 'params.numScreens');
validateattributes(seed, {'numeric'}, {'scalar', 'integer', 'nonnegative', 'finite'}, ...
    mfilename, 'seed');

[height, width] = size(inputField);
if height ~= width || mod(height, 2) ~= 0
    error('ao:InvalidGrid', 'inputField must be a square array with an even side length.');
end

previousRng = rng;
rngCleanup = onCleanup(@() rng(previousRng));
rng(seed, 'twister');

samplePitch = params.screenSize / height;
segmentLength = params.z / params.numScreens;
coordinate = (-height/2:height/2-1) * samplePitch;
[X, Y] = meshgrid(coordinate, coordinate);

outputField = inputField;
diagnostics.screenApplied = false(1, params.numScreens);
diagnostics.phaseRms = zeros(1, params.numScreens);
diagnostics.inputEnergy = sum(abs(inputField).^2, 'all');

for screenIndex = 1:params.numScreens
    phase = ao_von_karman_phase_screen(height, Cn2, segmentLength, params, X, Y);
    outputField = outputField .* exp(1i * phase);
    diagnostics.screenApplied(screenIndex) = true;
    diagnostics.phaseRms(screenIndex) = sqrt(mean(phase.^2, 'all'));
    outputField = ao_fresnel_propagate( ...
        outputField, params.lambda, segmentLength, samplePitch);
end

diagnostics.outputEnergy = sum(abs(outputField).^2, 'all');
diagnostics.relativeEnergyError = abs(diagnostics.outputEnergy ...
    - diagnostics.inputEnergy) / diagnostics.inputEnergy;
end
