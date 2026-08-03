function [separation, structureFunction] = ao_phase_structure_function(phase, samplePitch, maxLag)
%AO_PHASE_STRUCTURE_FUNCTION Estimate horizontal/vertical phase increments.

validateattributes(phase, {'numeric'}, {'2d', 'nonempty', 'real', 'finite'}, ...
    mfilename, 'phase');
validateattributes(samplePitch, {'numeric'}, {'scalar', 'real', 'positive', 'finite'}, ...
    mfilename, 'samplePitch');
validateattributes(maxLag, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
    mfilename, 'maxLag');
if maxLag >= min(size(phase))
    error('ao:InvalidLag', 'maxLag must be smaller than both phase dimensions.');
end

structureFunction = zeros(1, maxLag);
for lag = 1:maxLag
    horizontalDifference = phase(:, 1+lag:end) - phase(:, 1:end-lag);
    verticalDifference = phase(1+lag:end, :) - phase(1:end-lag, :);
    structureFunction(lag) = 0.5 * ( ...
        mean(horizontalDifference.^2, 'all') ...
        + mean(verticalDifference.^2, 'all'));
end
separation = (1:maxLag) * samplePitch;
end
