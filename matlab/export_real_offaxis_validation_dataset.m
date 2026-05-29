function export_real_offaxis_validation_dataset(outputPath, baseFolder, maxFramesPerTemperature, outputSize)
%EXPORT_REAL_OFFAXIS_VALIDATION_DATASET Export real off-axis hologram fields.
%
% This script follows the reconstruction pipeline used by:
%   E:\加扩束镜\文件处理\offaxisholo.m
%   E:\加扩束镜\文件处理\compute_offaxis_temperature_series.m
%
% It crops the first-order spectrum, center-pads it, reconstructs the complex
% optical field, and exports intensity/phase HDF5 data for Python validation.
%
% Example:
%   export_real_offaxis_validation_dataset( ...
%       fullfile('data', 'real_validation', 'real_offaxis_2x_validation.h5'), ...
%       'E:\加扩束镜\2倍放大', 200, [256 256])

if nargin < 1 || isempty(outputPath)
    outputPath = fullfile('data', 'real_validation', 'real_offaxis_2x_validation.h5');
end
if nargin < 2 || isempty(baseFolder)
    baseFolder = 'E:\加扩束镜\2倍放大';
end
if nargin < 3 || isempty(maxFramesPerTemperature)
    maxFramesPerTemperature = 200;
end
if nargin < 4 || isempty(outputSize)
    outputSize = [256 256];
end

cropRect = [179 72 236 120];
cropMode = "auto";
autoSearchRadius = 40;
autoDetectSampleCount = 5;

temperatures = discover_temperatures(baseFolder);
samplePlan = build_sample_plan(baseFolder, temperatures, maxFramesPerTemperature);
sampleCount = numel(samplePlan);
if sampleCount == 0
    error("realExport:NoSamples", "No PNG samples found under %s.", baseFolder);
end

outputDir = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
if exist(outputPath, 'file')
    delete(outputPath);
end

height = outputSize(1);
width = outputSize(2);
h5create(outputPath, '/input/intensity_turb', [height width sampleCount], ...
    'Datatype', 'single', 'ChunkSize', [height width 1]);
h5create(outputPath, '/input/phase_turb', [height width sampleCount], ...
    'Datatype', 'single', 'ChunkSize', [height width 1]);
h5create(outputPath, '/meta/frame_id', [sampleCount 1], 'Datatype', 'int64');
h5create(outputPath, '/meta/temperature', [sampleCount 1], 'Datatype', 'single');
h5create(outputPath, '/meta/crop_rect', [sampleCount 4], 'Datatype', 'int32');

frameIds = zeros(sampleCount, 1, 'int64');
temperatureValues = zeros(sampleCount, 1, 'single');
cropRectValues = zeros(sampleCount, 4, 'int32');
writeIndex = 1;

for temperatureIndex = 1:numel(temperatures)
    temperature = temperatures(temperatureIndex);
    rows = find([samplePlan.temperature] == temperature);
    imageFiles = [samplePlan(rows).path];
    firstImage = read_grayscale_image(imageFiles(1));
    [Ny, Nx] = size(firstImage);
    effectiveCropRect = resolve_crop_rect(imageFiles, cropRect, cropMode, ...
        autoSearchRadius, autoDetectSampleCount, Nx, Ny);

    fprintf("Exporting 温差 %g with crop [%d %d %d %d], %d frames\n", ...
        temperature, effectiveCropRect, numel(rows));

    for rowIndex = 1:numel(rows)
        sample = samplePlan(rows(rowIndex));
        hologram = read_grayscale_image(sample.path);
        reconstructedField = reconstruct_offaxis_field(hologram, effectiveCropRect);
        resizedField = resize_complex_field(reconstructedField, outputSize);
        intensity = single(abs(resizedField) .^ 2);
        phase = single(angle(resizedField));

        h5write(outputPath, '/input/intensity_turb', intensity, [1 1 writeIndex], [height width 1]);
        h5write(outputPath, '/input/phase_turb', phase, [1 1 writeIndex], [height width 1]);
        frameIds(writeIndex) = int64(sample.frameId);
        temperatureValues(writeIndex) = single(temperature);
        cropRectValues(writeIndex, :) = int32(effectiveCropRect);
        writeIndex = writeIndex + 1;
    end
end

h5write(outputPath, '/meta/frame_id', frameIds);
h5write(outputPath, '/meta/temperature', temperatureValues);
h5write(outputPath, '/meta/crop_rect', cropRectValues);
h5writeatt(outputPath, '/', 'description', 'Real 2x off-axis hologram complex-field validation dataset');
h5writeatt(outputPath, '/', 'source_base_folder', char(baseFolder));
h5writeatt(outputPath, '/', 'reconstruction_reference', 'offaxisholo.m / compute_offaxis_temperature_series.m');
fprintf("Wrote %d real validation frames to %s\n", sampleCount, outputPath);
end

function temperatures = discover_temperatures(baseFolder)
folderListing = dir(baseFolder);
folderListing = folderListing([folderListing.isdir]);
folderNames = setdiff(string({folderListing.name}), [".", ".."]);
temperatures = nan(1, numel(folderNames));
for folderIndex = 1:numel(folderNames)
    temperatureText = regexp(char(folderNames(folderIndex)), "\d+", "match", "once");
    temperatures(folderIndex) = str2double(temperatureText);
end
temperatures = sort(temperatures(~isnan(temperatures)));
end

function samplePlan = build_sample_plan(baseFolder, temperatures, maxFramesPerTemperature)
samplePlan = struct('path', strings(0), 'temperature', {}, 'frameId', {});
for temperature = temperatures
    folderPath = fullfile(baseFolder, "温差" + string(temperature));
    imageListing = sort_image_listing(dir(fullfile(folderPath, '*.png')));
    if ~isempty(maxFramesPerTemperature)
        imageListing = imageListing(1:min(maxFramesPerTemperature, numel(imageListing)));
    end
    for imageIndex = 1:numel(imageListing)
        imagePath = string(fullfile(imageListing(imageIndex).folder, imageListing(imageIndex).name));
        frameText = regexp(imageListing(imageIndex).name, "\d+", "match", "once");
        samplePlan(end + 1).path = imagePath; %#ok<AGROW>
        samplePlan(end).temperature = temperature;
        samplePlan(end).frameId = str2double(frameText);
    end
end
end

function imageListing = sort_image_listing(imageListing)
fileNames = string({imageListing.name});
numericNames = nan(size(fileNames));
for fileIndex = 1:numel(fileNames)
    [~, stem] = fileparts(fileNames(fileIndex));
    numericNames(fileIndex) = str2double(stem);
end
[~, order] = sortrows([numericNames(:), (1:numel(fileNames))']);
imageListing = imageListing(order);
end

function cropRect = resolve_crop_rect(imageFiles, referenceCropRect, cropMode, autoSearchRadius, autoDetectSampleCount, Nx, Ny)
validate_crop_rect(referenceCropRect, Nx, Ny);
if cropMode == "fixed"
    cropRect = referenceCropRect;
    return
end
[cropRect, detected] = detect_auto_crop_rect(imageFiles, referenceCropRect, ...
    autoSearchRadius, autoDetectSampleCount, Nx, Ny);
if ~detected
    warning("realExport:AutoCropFallback", ...
        "Auto crop detection failed; using fixed crop rectangle.");
    cropRect = referenceCropRect;
end
end

function [autoCropRect, detected] = detect_auto_crop_rect(imageFiles, referenceCropRect, autoSearchRadius, autoDetectSampleCount, Nx, Ny)
sampleCount = min(autoDetectSampleCount, numel(imageFiles));
averageSpectrum = zeros(Ny, Nx);
for sampleIndex = 1:sampleCount
    hologram = read_grayscale_image(imageFiles(sampleIndex));
    spectrum = fftshift(fft2(double(hologram)));
    averageSpectrum = averageSpectrum + log1p(abs(spectrum));
end
averageSpectrum = averageSpectrum ./ sampleCount;

[X, Y] = meshgrid(1:Nx, 1:Ny);
dcX = floor(Nx / 2) + 1;
dcY = floor(Ny / 2) + 1;
dcRadius = max(3, ceil(min(Nx, Ny) * 0.03));
dcMask = (X - dcX) .^ 2 + (Y - dcY) .^ 2 <= dcRadius ^ 2;
referenceCenterX = mean(referenceCropRect([1 3]));
referenceCenterY = mean(referenceCropRect([2 4]));
searchMask = (X - referenceCenterX) .^ 2 + ...
    (Y - referenceCenterY) .^ 2 <= autoSearchRadius ^ 2;
searchSpectrum = averageSpectrum;
searchSpectrum(dcMask | ~searchMask) = 0;
[peakValue, peakIndex] = max(searchSpectrum(:));
if ~isfinite(peakValue) || peakValue <= 0
    autoCropRect = referenceCropRect;
    detected = false;
    return
end

[peakY, peakX] = ind2sub([Ny, Nx], peakIndex);
cropWidth = referenceCropRect(3) - referenceCropRect(1) + 1;
cropHeight = referenceCropRect(4) - referenceCropRect(2) + 1;
x1 = round(peakX - (cropWidth - 1) / 2);
y1 = round(peakY - (cropHeight - 1) / 2);
autoCropRect = [x1, y1, x1 + cropWidth - 1, y1 + cropHeight - 1];
detected = autoCropRect(1) >= 1 && autoCropRect(2) >= 1 && ...
    autoCropRect(3) <= Nx && autoCropRect(4) <= Ny;
if ~detected
    autoCropRect = referenceCropRect;
end
end

function beamImage = read_grayscale_image(imagePath)
beamImage = imread(imagePath);
if ndims(beamImage) == 3
    beamImage = rgb2gray(beamImage);
end
end

function validate_crop_rect(cropRect, Nx, Ny)
if cropRect(1) > cropRect(3) || cropRect(2) > cropRect(4) || ...
        cropRect(3) > Nx || cropRect(4) > Ny
    error("realExport:InvalidCropRect", ...
        "CropRect [%d %d %d %d] is outside an image of size %d-by-%d.", ...
        cropRect(1), cropRect(2), cropRect(3), cropRect(4), Ny, Nx);
end
end

function reconstructedField = reconstruct_offaxis_field(hologram, cropRect)
[Ny, Nx] = size(hologram);
hologramSpectrum = fftshift(fft2(fftshift(double(hologram))));
croppedSpectrum = hologramSpectrum(cropRect(2):cropRect(4), cropRect(1):cropRect(3));
paddedSpectrum = zeros(Ny, Nx);
[cropHeight, cropWidth] = size(croppedSpectrum);
yStart = floor((Ny - cropHeight) / 2) + 1;
xStart = floor((Nx - cropWidth) / 2) + 1;
paddedSpectrum(yStart:yStart + cropHeight - 1, ...
    xStart:xStart + cropWidth - 1) = croppedSpectrum;
reconstructedField = ifftshift(ifft2(ifftshift(paddedSpectrum)));
end

function resizedField = resize_complex_field(field, outputSize)
if isequal(size(field), outputSize)
    resizedField = field;
    return
end
resizedField = imresize(real(field), outputSize, 'bilinear') + ...
    1i * imresize(imag(field), outputSize, 'bilinear');
end
