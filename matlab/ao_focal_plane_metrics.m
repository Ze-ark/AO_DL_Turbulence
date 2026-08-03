function [metrics, focalIntensity] = ao_focal_plane_metrics(pupilField, referenceField, bucketRadius)
%AO_FOCAL_PLANE_METRICS Compute normalized focal quality metrics.

if nargin < 3 || isempty(bucketRadius)
    bucketRadius = 2;
end
validateattributes(pupilField, {'numeric'}, {'2d', 'nonempty', 'finite'}, ...
    mfilename, 'pupilField');
if ~(isnumeric(referenceField) || islogical(referenceField)) ...
        || ~isequal(size(referenceField), size(pupilField)) ...
        || ~all(isfinite(double(referenceField)), 'all')
    error('ao:InvalidReferenceField', ...
        'referenceField must be a finite numeric or logical array matching pupilField.');
end
validateattributes(bucketRadius, {'numeric'}, {'scalar', 'real', 'nonnegative'}, ...
    mfilename, 'bucketRadius');

focalIntensity = abs(fftshift(fft2(ifftshift(pupilField)))).^2;
referenceIntensity = abs(fftshift(fft2(ifftshift(referenceField)))).^2;
normalizedPeak = max(focalIntensity, [], 'all') / sum(focalIntensity, 'all');
referencePeak = max(referenceIntensity, [], 'all') / sum(referenceIntensity, 'all');

N = size(pupilField, 1);
pixel = (-N/2:N/2-1);
[PX, PY] = meshgrid(pixel, pixel);
bucketMask = hypot(PX, PY) <= bucketRadius;

metrics.strehl = normalizedPeak / referencePeak;
metrics.powerInBucket = sum(focalIntensity(bucketMask), 'all') / sum(focalIntensity, 'all');
metrics.peakIntensity = max(focalIntensity, [], 'all');
end
