function [appliedPhase, nextState, diagnostics] = ao_slm_step(state, requestedPhase, limits)
%AO_SLM_STEP Apply phase range, quantization, delay, and slew constraints.
%
% This is a hardware-independent S0 execution model. Measured FSLM parameters
% will replace the assumed limits during the later sim-to-real stage.

validateattributes(requestedPhase, {'numeric'}, {'2d', 'real', 'finite'}, ...
    mfilename, 'requestedPhase');
validateattributes(limits.phaseMin, {'numeric'}, {'scalar', 'real', 'finite'}, ...
    mfilename, 'limits.phaseMin');
validateattributes(limits.phaseMax, {'numeric'}, {'scalar', 'real', 'finite'}, ...
    mfilename, 'limits.phaseMax');
validateattributes(limits.maxDelta, {'numeric'}, {'scalar', 'real', 'nonnegative'}, ...
    mfilename, 'limits.maxDelta');
validateattributes(limits.quantizationLevels, {'numeric'}, ...
    {'scalar', 'integer', 'nonnegative'}, mfilename, 'limits.quantizationLevels');
validateattributes(limits.delayFrames, {'numeric'}, {'scalar', 'integer', 'nonnegative'}, ...
    mfilename, 'limits.delayFrames');
if limits.phaseMax <= limits.phaseMin
    error('ao:InvalidSlmLimits', 'phaseMax must be greater than phaseMin.');
end

nextState = initialize_state(state, size(requestedPhase), limits.delayFrames);
clipped = min(max(requestedPhase, limits.phaseMin), limits.phaseMax);
saturated = clipped ~= requestedPhase;
quantized = quantize_phase(clipped, limits);
[delayedCommand, nextState.commandQueue] = apply_delay( ...
    quantized, nextState.commandQueue, limits.delayFrames);

requestedDelta = delayedCommand - nextState.currentPhase;
limitedDelta = min(max(requestedDelta, -limits.maxDelta), limits.maxDelta);
appliedPhase = nextState.currentPhase + limitedDelta;
nextState.currentPhase = appliedPhase;

diagnostics.saturatedFraction = mean(saturated, 'all');
diagnostics.slewLimitedFraction = mean(limitedDelta ~= requestedDelta, 'all');
diagnostics.delayedCommand = delayedCommand;
end

function state = initialize_state(state, fieldSize, delayFrames)
if isempty(state)
    state.currentPhase = zeros(fieldSize);
    state.commandQueue = zeros([fieldSize, delayFrames]);
    return
end
if ~isequal(size(state.currentPhase), fieldSize)
    error('ao:InvalidSlmState', 'state.currentPhase must match requestedPhase.');
end
if ~isequal(size(state.commandQueue), [fieldSize, delayFrames])
    error('ao:InvalidSlmState', 'state.commandQueue does not match delayFrames.');
end
end

function quantized = quantize_phase(phase, limits)
if limits.quantizationLevels <= 1
    quantized = phase;
    return
end
step = (limits.phaseMax - limits.phaseMin) / (limits.quantizationLevels - 1);
quantized = limits.phaseMin + round((phase - limits.phaseMin) / step) * step;
end

function [delayed, queue] = apply_delay(command, queue, delayFrames)
if delayFrames == 0
    delayed = command;
    return
end
delayed = queue(:, :, 1);
queue = cat(3, queue(:, :, 2:end), command);
end
