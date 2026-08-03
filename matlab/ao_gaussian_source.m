function [field, X, Y] = ao_gaussian_source(N, params)
%AO_GAUSSIAN_SOURCE Create the source-plane Gaussian complex field.

validateattributes(N, {'numeric'}, {'scalar', 'integer', 'even', '>=', 4}, ...
    mfilename, 'N');
validateattributes(params.screenSize, {'numeric'}, {'scalar', 'real', 'positive'}, ...
    mfilename, 'params.screenSize');
validateattributes(params.w0, {'numeric'}, {'scalar', 'real', 'positive'}, ...
    mfilename, 'params.w0');

delta = params.screenSize / N;
x = (-N/2:N/2-1) * delta;
[X, Y] = meshgrid(x, x);
field = exp(-(X.^2 + Y.^2) ./ params.w0^2);
end
