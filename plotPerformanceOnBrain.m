function perfData = plotPerformanceOnBrain(filepath)

data = load(filepath);
perfData = data.perfData;
config = data.config;
    
% This script generates brain images were areas of different networks are
% colored according to some performance measure (e.g., effect size;
% correlation with some model; decoding accuracy; etc.)
%
% INPUT:
% perfData = structure array, with one field per network.
%            Each field is a 1xN array of performance values.
%            currently supported fields: lang, md, dmn
% config = structure array with configuration info. Fields:
%          minMax = 1x2 number array, minimum and maximum performance values (for scaling the plot colors)
%          theMap = either a string with a known colormap name (e.g., 'jet'),
%                   or a Nx3 array of RGB values defining a colormap
%          colorWeight = number between 0 and 1, percentage saturation of performance color
%                        relative to the background brain image (a good default is 0.25)
%          measureName = string, name of measure (e.g., 'accuracy')
%
% OUTPUT:
% For each network, a subplot of lateral and/or medial brain images,
% with network areas colored according to perfData
%
% NOTE:
% The regions depicted are "masks" based on group-level functional data;
% they only show where subject-specific functional network regions are likely to fall,
% but the actual functional regions are both smaller and highly variable in
% their precise locations across individuals.
%
% Idan Blank, September 2018

load networkImages
netNames = fieldnames(perfData);
nNets = numel(netNames);
bgIms = {'Lateral', 'Medial'};      % names of backgroud images
nBg = numel(bgIms);

if ischar(config.theMap)
    clrMap = colormap(config.theMap)*255;
else
    clrMap = config.theMap;
    if max(clrMap(:)) <= 1
        clrMap = clrMap*255;
    end
end
nColors = size(clrMap,1);
cw = config.colorWeight;

figure(1)
clf reset
% set(gcf, 'units', 'normalized', 'position', [0.05 0.05 0.9 0.9]);
set(gcf, 'units', 'normalized', 'position', [0.05 0.05 0.9 0.85]);

for nInd = 1:nNets
    %% Find the number of LH regions for current network, and which brain images (lateral / medial) need to be plotted %%
    nRegions = zeros(nBg,1);
    isPlot = false(nBg,1);
    for ii = 1:nBg
        currName = [netNames{nInd}, 'Masks', bgIms{ii}];
        if exist(currName, 'var')
            eval(['nRegions(ii) = max(', currName, '(:));']);
            isPlot(ii) = true;
        end
    end
    nRegions = max(nRegions);

    %% Generate subplots for each brain image %%
    for ii = 1:nBg
        if isPlot(ii)
            eval(['masks = ', netNames{nInd}, 'Masks', bgIms{ii}, ';']);            
            eval(['imLH = ', bgIms{ii}, ';']);
            imLH = imLH; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% /255;
            imRH = fliplr(imLH);
            imHemis = cat(3, imLH, imRH);          
            
            %% Loop over LH and RH %%
            for hInd = 1:2
                im = imHemis(:,:,hInd);
                imR = im;
                imG = im;
                imB = im;

                if hInd == 2
                    masks = fliplr(masks) + nRegions;
                    masks(masks==nRegions) = 0;
                end
                
                maskInds = setdiff(unique(masks(:)),0);                          
                for mInd = maskInds'
                    %% Convert current performance value into RGB
                    value = perfData.(netNames{nInd})(mInd)-config.minMax(1);
                    if value == -1 % not significant
                        clrMapRGB = [192,192,192]; % gray
                    else
                        clrMapPrcnt = value/(config.minMax(2)-config.minMax(1));  % relative location (%) of current performance value along the config.minMax scale
                        clrMapRowInterp = clrMapPrcnt*nColors;                                         % row index in clrMap to be interpolated (e.g., 2.7 will interpolate between rows 2 and 3)
                        clrMapRow1 = max(floor(clrMapRowInterp),1);
                        clrMapRow2 = min(ceil(clrMapRowInterp),nColors);
                        if clrMapRow1 < clrMapRow2
                            clrMapRGB = interp1([clrMapRow1, clrMapRow2], clrMap([clrMapRow1, clrMapRow2],:), clrMapRowInterp); % linear interpolation
                        else
                            clrMapRGB = clrMap(clrMapRow1,:);
                        end
                    end
                    
                    %% Fill in the location on the brain image corresponding to the current mask with the color corresponding to the performance value %%
                    currMask = (masks == mInd);                    
                    imR(currMask) = (cw + (1-cw)*(imR(currMask)/max(max(imR(currMask)))))*clrMapRGB(1);
                    imG(currMask) = (cw + (1-cw)*(imG(currMask)/max(max(imG(currMask)))))*clrMapRGB(2);
                    imB(currMask) = (cw + (1-cw)*(imB(currMask)/max(max(imB(currMask)))))*clrMapRGB(3);
                    
                    %% Make a white contour around the mask %%
                    currBorder = conv2(double(currMask),(1/(4^2))*ones(4),'same');
                    currBorder = (currBorder > 0) & (currBorder < 1);                    
                                        
                    borderColor = im(currBorder);
                    borderColor = borderColor - min(borderColor);  % now, min = 0
                    borderColor = borderColor/max(borderColor);    % now, max = 1
                    borderColor = borderColor*105+150;             % now, min = 200, max = 255
                    imR(currBorder) = borderColor;
                    imG(currBorder) = borderColor;
                    imB(currBorder) = borderColor;

                end
                
                %% Plot %%
                im = cat(3,imR,imG,imB);
                clrMapIm = imresize(flipud(permute(clrMap,[1,3,2])), [size(im,1),0.05*size(im,2)]);
                im = cat(2, im, clrMapIm);   % this is a "colorbar" added at the right end of the image
                
                subplot(nNets, 2*nBg, (nInd-1)*2*nBg+(hInd-1)*nBg+ii);
                imagesc(im/255);
                set(gca, 'xColor', 'none', ...
                    'YAxisLocation', 'right', ...
                    'ytick', linspace(0.5, size(im,1)-0.5, 5), 'yticklabels', fliplr(linspace(config.minMax(1), config.minMax(2), 5)), ...
                    'fontname', 'arial', 'fontsize', 14);
                
                if (ii == 1) && (hInd == 1)
                    text(0-0.1*size(im,2), size(im,1)/2, upper(netNames{nInd}), 'fontname', 'arial', 'fontsize', 16, 'fontweight', 'bold', ...
                        'rotation', 90, 'horizontalalignment', 'center', 'verticalalignment',' middle');
                    if nInd == 1
                        ylabel(config.measureName, 'fontname', 'arial', 'fontsize', 14);
                    end                        
                end

            end
        end
    end
end

savefig([filepath(1:end-3), 'fig']);
saveas(gcf, [filepath(1:end-3), 'png']);
close(gcf);
