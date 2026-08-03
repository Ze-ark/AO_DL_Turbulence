function outputField = ao_fresnel_propagate(inputField, wavelength, distance, samplePitch)
%AO_FRESNEL_PROPAGATE Propagate a square field with a Fresnel transfer function.

validateattributes(inputField, {'numeric'}, {'2d', 'nonempty', 'finite'}, ...
    mfilename, 'inputField');
validateattributes(wavelength, {'numeric'}, {'scalar', 'real', 'positive', 'finite'}, ...
    mfilename, 'wavelength');
validateattributes(distance, {'numeric'}, {'scalar', 'real', 'nonnegative', 'finite'}, ...
    mfilename, 'distance');
validateattributes(samplePitch, {'numeric'}, {'scalar', 'real', 'positive', 'finite'}, ...
    mfilename, 'samplePitch');

[height, width] = size(inputField);
if height ~= width || mod(height, 2) ~= 0
    error('ao:InvalidGrid', 'inputField must be a square array with an even side length.');
end

frequency = (-height/2:height/2-1) / (height * samplePitch);
[FX, FY] = meshgrid(frequency, frequency);
transfer = exp(-1i * pi * wavelength * distance .* (FX.^2 + FY.^2));
spectrum = fftshift(fft2(ifftshift(inputField)));
outputField = fftshift(ifft2(ifftshift(spectrum .* transfer)));
end
