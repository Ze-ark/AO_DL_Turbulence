function simulate_gaussian_turbulence_dataset(outputPath, numFrames, N, params)
%SIMULATE_GAUSSIAN_TURBULENCE_DATASET Export S0 split-step Gaussian data.
%
% Both turbulent input and clean target are evaluated at the receiver plane.
% Each frame uses a recorded independent random seed. This static exporter is
% an S0 physics check; a later stage will create time-correlated RL episodes.

if nargin < 1 || isempty(outputPath)
    outputPath = fullfile('data', 'raw_matlab_exports', 'sim_gaussian_v1.h5');
end
if nargin < 2 || isempty(numFrames)
    numFrames = 1000;
end
if nargin < 3 || isempty(N)
    N = 256;
end
if nargin < 4 || isempty(params)
    params = ao_default_simulation_params();
end

validateattributes(numFrames, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
    mfilename, 'numFrames');
validateattributes(N, {'numeric'}, {'scalar', 'integer', 'even', '>=', 4}, ...
    mfilename, 'N');

outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if exist(outputPath, 'file')
    delete(outputPath);
end

[sourceField, ~, ~] = ao_gaussian_source(N, params);
[cleanReceiver, cleanDiagnostics] = ao_split_step_propagate( ...
    sourceField, 0, params, params.baseSeed);
cleanIntensity = single(abs(cleanReceiver).^2);
cleanPhase = single(angle(cleanReceiver));

shape = [N, N, numFrames];
h5create(outputPath, '/input/intensity_turb', shape, 'Datatype', 'single', 'ChunkSize', [N, N, 1]);
h5create(outputPath, '/input/phase_turb', shape, 'Datatype', 'single', 'ChunkSize', [N, N, 1]);
h5create(outputPath, '/target/intensity_clean', shape, 'Datatype', 'single', 'ChunkSize', [N, N, 1]);
h5create(outputPath, '/target/phase_clean', shape, 'Datatype', 'single', 'ChunkSize', [N, N, 1]);
h5create(outputPath, '/meta/frame_id', [numFrames, 1], 'Datatype', 'int64');
h5create(outputPath, '/meta/scene_id', [numFrames, 1], 'Datatype', 'int64');
h5create(outputPath, '/meta/random_seed', [numFrames, 1], 'Datatype', 'int64');
h5create(outputPath, '/meta/turbulence_strength', [numFrames, 1], 'Datatype', 'single');
h5create(outputPath, '/meta/r0', [numFrames, 1], 'Datatype', 'single');
h5create(outputPath, '/meta/r0_or_equivalent', [numFrames, 1], 'Datatype', 'single');

frameIds = int64((0:numFrames-1)');
randomSeeds = int64(params.baseSeed + (0:numFrames-1)');
strengths = single(linspace(0.2, 1.0, numFrames)');
r0Values = zeros(numFrames, 1, 'single');

for frame = 1:numFrames
    Cn2 = params.Cn2Base * double(strengths(frame));
    turbulentReceiver = ao_split_step_propagate( ...
        sourceField, Cn2, params, double(randomSeeds(frame)));
    r0Values(frame) = single(ao_fried_parameter(Cn2, params.lambda, params.z));

    h5write(outputPath, '/input/intensity_turb', single(abs(turbulentReceiver).^2), [1, 1, frame], [N, N, 1]);
    h5write(outputPath, '/input/phase_turb', single(angle(turbulentReceiver)), [1, 1, frame], [N, N, 1]);
    h5write(outputPath, '/target/intensity_clean', cleanIntensity, [1, 1, frame], [N, N, 1]);
    h5write(outputPath, '/target/phase_clean', cleanPhase, [1, 1, frame], [N, N, 1]);
end

h5write(outputPath, '/meta/frame_id', frameIds);
h5write(outputPath, '/meta/scene_id', frameIds);
h5write(outputPath, '/meta/random_seed', randomSeeds);
h5write(outputPath, '/meta/turbulence_strength', strengths);
h5write(outputPath, '/meta/r0', r0Values);
h5write(outputPath, '/meta/r0_or_equivalent', r0Values);
h5writeatt(outputPath, '/', 'description', 'S0 split-step Gaussian beam turbulence dataset');
h5writeatt(outputPath, '/', 'scientific_gate', 'S0_physics_passed_dynamic_pending');
h5writeatt(outputPath, '/', 'target_plane', 'receiver');
h5writeatt(outputPath, '/', 'r0_definition', 'full-path plane-wave Fried parameter');
h5writeatt(outputPath, '/', 'propagation', 'spatial phase screen followed by Fresnel propagation per segment');
h5writeatt(outputPath, '/', 'clean_relative_energy_error', cleanDiagnostics.relativeEnergyError);
h5writeatt(outputPath, '/', 'layout', 'Arrays are [height, width, frame] in MATLAB.');
fprintf('Wrote %d S0 frames to %s\n', numFrames, outputPath);
end
