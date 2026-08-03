function params = ao_default_simulation_params()
%AO_DEFAULT_SIMULATION_PARAMS Return the S0 optical propagation parameters.

params.z = 3300;                  % Total propagation distance, m
params.w0 = 0.012;                % Gaussian beam waist radius, m
params.Cn2Base = 5e-15;           % Refractive-index structure constant, m^(-2/3)
params.lambda = 1550e-9;          % Wavelength, m
params.numScreens = 8;            % Number of split-step phase screens
params.L0 = 10;                   % Turbulence outer scale, m
params.l0 = 0.01;                 % Turbulence inner scale, m
params.screenSize = 1.0;          % Transverse grid width, m
params.subharmonicLevels = 3;     % Low-frequency compensation levels
params.baseSeed = 42;             % First independent scene seed
end
