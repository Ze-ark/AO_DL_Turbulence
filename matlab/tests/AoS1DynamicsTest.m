classdef AoS1DynamicsTest < matlab.unittest.TestCase
    %AOS1DYNAMICSTEST Tests for Taylor frozen-flow temporal evolution.

    methods (TestClassSetup)
        function addMatlabFolder(testCase)
            testsFolder = fileparts(mfilename('fullpath'));
            matlabFolder = fileparts(testsFolder);
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(matlabFolder));
        end
    end

    methods (Test)
        function integerShiftMatchesCircularTranslation(testCase)
            phase = reshape(1:256, 16, 16);
            samplePitch = 0.01;

            actual = ao_taylor_frozen_flow_step( ...
                phase, samplePitch, 2 * samplePitch, -samplePitch, 1);
            expected = circshift(phase, [-1, 2]);
            expected = expected - mean(expected, 'all');

            testCase.verifyEqual(actual, expected, 'AbsTol', 2e-12);
        end

        function fractionalFrozenShiftPreservesPhaseVariance(testCase)
            coordinate = (0:31) / 32;
            [X, Y] = meshgrid(coordinate, coordinate);
            phase = sin(2 * pi .* X) + 0.4 .* cos(4 * pi .* Y);

            shifted = ao_taylor_frozen_flow_step(phase, 1, 0.37, -0.23, 1);

            testCase.verifyEqual(std(shifted, 1, 'all'), std(phase, 1, 'all'), ...
                'AbsTol', 2e-12);
        end

        function zeroPersistenceReturnsInnovationWithoutPiston(testCase)
            phase = zeros(16);
            innovation = reshape(sin(1:256), 16, 16);
            expected = innovation - mean(innovation, 'all');

            actual = ao_taylor_frozen_flow_step(phase, 1, 0.2, 0.3, 0, innovation);

            testCase.verifyEqual(actual, expected, 'AbsTol', 2e-12);
        end
    end
end
