function r0 = ao_fried_parameter(Cn2, wavelength, pathLength)
%AO_FRIED_PARAMETER Compute the plane-wave Fried parameter for a full path.
%
% For constant Cn2 along the path:
%   r0 = (0.423 * k^2 * Cn2 * pathLength)^(-3/5)

validateattributes(Cn2, {'numeric'}, {'scalar', 'real', 'nonnegative', 'finite'}, ...
    mfilename, 'Cn2');
validateattributes(wavelength, {'numeric'}, {'scalar', 'real', 'positive', 'finite'}, ...
    mfilename, 'wavelength');
validateattributes(pathLength, {'numeric'}, {'scalar', 'real', 'positive', 'finite'}, ...
    mfilename, 'pathLength');

if Cn2 == 0
    r0 = Inf;
    return
end

k = 2 * pi / wavelength;
r0 = (0.423 * k^2 * Cn2 * pathLength)^(-3/5);
end
