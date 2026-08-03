function [rmse, piston] = ao_masked_phase_rmse(estimatedPhase, targetPhase, pupilMask)
%AO_MASKED_PHASE_RMSE Measure wrapped phase error inside a pupil without piston.

validateattributes(estimatedPhase, {'numeric'}, {'2d', 'finite'}, mfilename, 'estimatedPhase');
validateattributes(targetPhase, {'numeric'}, {'size', size(estimatedPhase), 'finite'}, ...
    mfilename, 'targetPhase');
validateattributes(pupilMask, {'logical', 'numeric'}, {'size', size(estimatedPhase)}, ...
    mfilename, 'pupilMask');
mask = logical(pupilMask);
if ~any(mask, 'all')
    error('ao:EmptyPupil', 'pupilMask must contain at least one valid pixel.');
end

wrappedError = angle(exp(1i .* (estimatedPhase - targetPhase)));
piston = angle(sum(exp(1i .* wrappedError(mask))));
residual = angle(exp(1i .* (wrappedError(mask) - piston)));
rmse = sqrt(mean(residual.^2));
end
