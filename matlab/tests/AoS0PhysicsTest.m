classdef AoS0PhysicsTest < matlab.unittest.TestCase
    %AOS0PHYSICSTEST Unit tests for the S0 optical propagation gate.

    properties (TestParameter)
        Seed = {7, 42, 2026}
    end

    methods (TestClassSetup)
        function addMatlabFolder(testCase)
            testsFolder = fileparts(mfilename('fullpath'));
            matlabFolder = fileparts(testsFolder);
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(matlabFolder));
        end
    end

    methods (Test)
        function zeroTurbulenceMatchesOneFullPropagation(testCase)
            params = AoS0PhysicsTest.smallParams();
            [source, ~, ~] = ao_gaussian_source(64, params);
            expected = ao_fresnel_propagate(source, params.lambda, ...
                params.z, params.screenSize / 64);

            actual = ao_split_step_propagate(source, 0, params, 42);

            testCase.verifyEqual(actual, expected, 'AbsTol', 2e-12);
        end

        function splitStepConservesEnergy(testCase, Seed)
            params = AoS0PhysicsTest.smallParams();
            [source, ~, ~] = ao_gaussian_source(64, params);

            [~, diagnostics] = ao_split_step_propagate( ...
                source, params.Cn2Base, params, Seed);

            testCase.verifyLessThan(diagnostics.relativeEnergyError, 1e-12);
        end

        function firstAndEveryScreenAreApplied(testCase)
            params = AoS0PhysicsTest.smallParams();
            params.numScreens = 1;
            [source, ~, ~] = ao_gaussian_source(64, params);
            clean = ao_split_step_propagate(source, 0, params, 11);

            [turbulent, diagnostics] = ao_split_step_propagate( ...
                source, params.Cn2Base, params, 11);

            testCase.verifyTrue(all(diagnostics.screenApplied));
            testCase.verifyGreaterThan(diagnostics.phaseRms, 0);
            testCase.verifyGreaterThan(norm(turbulent - clean, 'fro'), 1e-6);
        end

        function seedsAreReproducibleAndIndependent(testCase)
            params = AoS0PhysicsTest.smallParams();
            [source, ~, ~] = ao_gaussian_source(48, params);
            first = ao_split_step_propagate(source, params.Cn2Base, params, 17);

            repeated = ao_split_step_propagate(source, params.Cn2Base, params, 17);
            different = ao_split_step_propagate(source, params.Cn2Base, params, 18);

            testCase.verifyEqual(repeated, first, 'AbsTol', 0);
            testCase.verifyGreaterThan(norm(different - first, 'fro'), 1e-6);
        end

        function friedParameterUsesFullPath(testCase)
            params = AoS0PhysicsTest.smallParams();
            k = 2 * pi / params.lambda;
            expected = (0.423 * k^2 * params.Cn2Base * params.z)^(-3/5);

            actual = ao_fried_parameter(params.Cn2Base, params.lambda, params.z);

            testCase.verifyEqual(actual, expected, 'RelTol', 10 * eps);
        end

        function exporterStoresReceiverPlaneReference(testCase)
            params = AoS0PhysicsTest.smallParams();
            outputPath = [tempname, '.h5'];
            testCase.addTeardown(@delete, outputPath);
            [source, ~, ~] = ao_gaussian_source(32, params);
            expected = ao_split_step_propagate(source, 0, params, params.baseSeed);

            simulate_gaussian_turbulence_dataset(outputPath, 2, 32, params);
            storedIntensity = double(h5read(outputPath, '/target/intensity_clean', [1, 1, 1], [32, 32, 1]));
            storedPhase = double(h5read(outputPath, '/target/phase_clean', [1, 1, 1], [32, 32, 1]));

            testCase.verifyEqual(storedIntensity, abs(expected).^2, 'AbsTol', 2e-7);
            testCase.verifyEqual(exp(1i * storedPhase), exp(1i * angle(expected)), 'AbsTol', 2e-7);
            testCase.verifyGreaterThan(norm(storedIntensity - abs(source).^2, 'fro'), 1e-4);
        end

        function knownDefocusCorrectionImprovesFocus(testCase)
            N = 64;
            coordinate = linspace(-1, 1, N);
            [X, Y] = meshgrid(coordinate, coordinate);
            pupil = hypot(X, Y) <= 0.8;
            defocus = 5 .* (X.^2 + Y.^2) .* pupil;
            aberrated = pupil .* exp(1i .* defocus);
            corrected = aberrated .* exp(-1i .* defocus);

            before = ao_focal_plane_metrics(aberrated, pupil);
            after = ao_focal_plane_metrics(corrected, pupil);

            testCase.verifyGreaterThan(after.strehl, before.strehl);
            testCase.verifyEqual(after.strehl, 1, 'AbsTol', 1e-12);
            testCase.verifyGreaterThan(after.powerInBucket, before.powerInBucket);
        end

        function phaseScreenHasKolmogorovStructureFunction(testCase)
            [separation, measured, expected] = ...
                AoS0PhysicsTest.ensembleStructureFunction();
            fitRange = 2:8;
            coefficients = polyfit(log(separation(fitRange)), ...
                log(measured(fitRange)), 1);
            amplitudeRatio = median(measured(fitRange) ./ expected(fitRange));

            testCase.verifyGreaterThan(coefficients(1), 1.15);
            testCase.verifyLessThan(coefficients(1), 2.15);
            testCase.verifyGreaterThan(amplitudeRatio, 0.35);
            testCase.verifyLessThan(amplitudeRatio, 2.8);
        end

        function slmRangeAndSlewLimitsAreEnforced(testCase)
            limits = AoS0PhysicsTest.slmLimits();
            limits.delayFrames = 0;
            limits.maxDelta = 0.25;
            requested = 2 * ones(8);

            [applied, ~, diagnostics] = ao_slm_step([], requested, limits);

            testCase.verifyEqual(applied, 0.25 * ones(8), 'AbsTol', 0);
            testCase.verifyEqual(diagnostics.saturatedFraction, 1);
            testCase.verifyEqual(diagnostics.slewLimitedFraction, 1);
        end

        function slmDelayUsesCommandsOnlyAfterConfiguredFrames(testCase)
            limits = AoS0PhysicsTest.slmLimits();
            limits.delayFrames = 2;
            command = 0.5 * ones(4);

            [first, state] = ao_slm_step([], command, limits);
            [second, state] = ao_slm_step(state, command, limits);
            [third, ~] = ao_slm_step(state, command, limits);

            testCase.verifyEqual(first, zeros(4), 'AbsTol', 0);
            testCase.verifyEqual(second, zeros(4), 'AbsTol', 0);
            testCase.verifyEqual(third, command, 'AbsTol', 0);
        end

        function slmPhaseIsQuantizedToConfiguredLevels(testCase)
            limits = AoS0PhysicsTest.slmLimits();
            limits.quantizationLevels = 5;
            requested = [-0.74, -0.26, 0.10, 0.74];

            applied = ao_slm_step([], requested, limits);

            testCase.verifyEqual(applied, [-0.5, -0.5, 0, 0.5], 'AbsTol', 0);
        end

        function maskedPhaseErrorIgnoresPistonAndOutsidePupil(testCase)
            mask = false(16);
            mask(5:12, 5:12) = true;
            target = zeros(16);
            estimated = 0.7 * ones(16);
            estimated(~mask) = 2.4;

            [rmse, piston] = ao_masked_phase_rmse(estimated, target, mask);

            testCase.verifyEqual(rmse, 0, 'AbsTol', 1e-12);
            testCase.verifyEqual(piston, 0.7, 'AbsTol', 1e-12);
        end

        function invalidGridHasClearIdentifier(testCase)
            invalidField = ones(31, 32);

            action = @() ao_fresnel_propagate(invalidField, 1550e-9, 10, 1e-3);

            testCase.verifyError(action, 'ao:InvalidGrid');
        end
    end

    methods (Static, Access = private)
        function params = smallParams()
            params = ao_default_simulation_params();
            params.z = 100;
            params.w0 = 0.08;
            params.screenSize = 0.4;
            params.numScreens = 3;
            params.subharmonicLevels = 2;
        end

        function limits = slmLimits()
            limits.phaseMin = -1;
            limits.phaseMax = 1;
            limits.maxDelta = Inf;
            limits.quantizationLevels = 0;
            limits.delayFrames = 0;
        end

        function [separation, measured, expected] = ensembleStructureFunction()
            params = ao_default_simulation_params();
            params.screenSize = 1;
            params.L0 = 100;
            params.l0 = 1e-3;
            params.subharmonicLevels = 0;
            N = 128;
            Cn2 = 1e-13;
            segmentLength = 100;
            samplePitch = params.screenSize / N;
            coordinate = (-N/2:N/2-1) * samplePitch;
            [X, Y] = meshgrid(coordinate, coordinate);
            measured = zeros(1, 12);
            previousRng = rng;
            cleanup = onCleanup(@() rng(previousRng));
            for seed = 1:24
                rng(seed, 'twister');
                phase = ao_von_karman_phase_screen( ...
                    N, Cn2, segmentLength, params, X, Y);
                [separation, oneEstimate] = ao_phase_structure_function( ...
                    phase, samplePitch, 12);
                measured = measured + oneEstimate;
            end
            measured = measured / 24;
            r0 = ao_fried_parameter(Cn2, params.lambda, segmentLength);
            expected = 6.88 .* (separation ./ r0).^(5/3);
        end
    end
end
