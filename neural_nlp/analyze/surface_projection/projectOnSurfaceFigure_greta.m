function projectOnSurfaceFigure_greta(fileNames, config, freeview)
    
% This scripts takes projects data onto an Freesurfer's average brain surface
% 
% INPUT:
% fileNames = f X 1 cell of strings, containing full paths to the files that should be projected
%             NOTE: these must be full paths, even if you are in the correct directory
% config = structure with the following fields:
%          doCoreg = boolean, whether files need to be coregistered to Freesurger's T1
%          interp = integer, interpolation method when registering data following SPM's CO-REGISTER parameters
%          hemi = string, either 'lh', or 'rh'
%          projectStr = string, method for mri_vol2surf, e.g., 'projfrac 0', (project on GM-WM border);
%                       'projfrac 1' (project on cortical surface), 'projdist 1' (project 1mm deep into the surface),
%                       'projfrac-max 0 1 0.05 (maximum value of samples along the normal to the surface in 5% increments)
%                       'projfrac-avg 0 1 0.05', etc.
%          surfaceFile = string, name of surface file for plotting from among Freesurfer's options
%                        (e.g., 'inflated_lh', 'pial_lh')
%          threshold = 1 X 2 double vector, lower and upper threshold values for coloring
%          (optional) azimuth = integer, camera azimuth in degrees (0 = left view, 180 = right view)
%          (optional) camStr = string with further visualization parameters for freeview's --cam option, e.g., 'elevation 30' (see freeview -h)
% 
% freeview = binary. If freeview == 1, saves the Freeview screenshot automatically
%
% Idan Blank, Aug 05 2019, edits: Greta Tuckute, January 2019

%% Parameters %%
SPM_T1_path = fullfile(pwd,'SPM_T1.nii');                % SPM's average T1
FS_T1_path = fullfile(pwd,'FreesurferT1.nii');           % FreeSurfer's average T1
subj_T1_path = fullfile(pwd,'forReg_ID231_T1_z69.nii');  % A random subject's T1

origFileNames = fileNames; % Takes the .nii file string, i.e. /mindhive/evlab/u/gretatu/Desktop/surface_projection/padded_brain_rand.nii
nFiles = numel(fileNames);

setupCommands = ['export FREESURFER_HOME=/software/Freesurfer/5.3.0; ', ...
    'export SUBJECTS_DIR=/software/Freesurfer/5.3.0/subjects; ', ...
    'source $FREESURFER_HOME/SetUpFreeSurfer.sh; '];

freeviewSurfaceDir = '/software/Freesurfer/5.3.0/subjects/cvs_avg35_inMNI152/surf/';

if ~isfield(config, 'camStr')
    config.camStr = '';
end
if ~isfield(config, 'azimuth')
    if strcmp(config.hemi, 'lh')
        config.azimuth = 0;
    else
        config.azimuth = 180;
    end
end

%% Make sure file formats are consistent %%
for f = 1:nFiles
    dotInds = strfind(fileNames{f}, '.'); % idx for .nii extension   
    currFormat = fileNames{f}((dotInds(end)+1):end); % current format: 'nii'
    % If the current format is NOT nii. does not go into the loop.
    if ~strcmp(currFormat, 'nii')
        % display('in loop')
        newFileName = [fileNames{f}(1:dotInds(end)), 'nii'];
        [status] = systems(['mri_convert ', fileNames{f}, newFileName])
        fileNames{f} = newFileName;
    end
end

%% Make sure dimensions are consistent across fileNames + subj_T1_path %%
fileSizes = nan(nFiles+1,3); % [2,3] matrix
for f = 1:nFiles
    volInfo = spm_vol(fileNames{f}); % Get the volume info from the nii file
    fileSizes(f,:) = volInfo.dim; % adds the file size as the first row
end
if config.doCoreg % Goes into loop
    volInfo = spm_vol(subj_T1_path); % [79,95,69]   
    fileNames = [fileNames; subj_T1_path]; % now filenames contain: 1st row: padded_brain.nii
    % 2nd row: the subj reg file.nii
else
    volInfo = spm_vol(FS_T1_path);
    fileNames = [fileNames; FS_T1_path];    
end
fileSizes(end,:) = volInfo.dim; % adds the FS subj reg file dimension in 2nd row

[uniqueRows,~,uniqueRowInds] = unique(fileSizes,'rows');
modeRowInd = mode(uniqueRowInds);              % index of row that appears the most in uniqueRowInds
% modeRowInd is 1.
modeRow = uniqueRows(modeRowInd,:);            % the file dimension that is most common (mode)
% The first one, ie.  79    95    69
badInds = find(~(uniqueRowInds==modeRowInd));  % indices of files whose size is not the most common
% badInds is 1

disp(['Making sure all files have the following dimensions: ', mat2str(modeRow)]);
% making sure files have the FS reg file dimension

for f = 1:numel(badInds) % f=1
    clear imcalcInfo    
    if isempty(find(badInds == nFiles+1, 1)) % goes into loop, =1
        imcalcInfo.input{1} = fileNames{end}; % adds this file as input forReg_ID231_T1_z69.nii
    else
        imcalcInfo.input{1} = fileNames{modeRowInd};        
    end
    imcalcInfo.input{2} = fileNames{badInds(f)}; % adds the padded.nii as the second input
    imcalcInfo.expression = 'i2 + i1 - i1';        % this will convert i2 to have the dimesnions of i1
    
    % ie it did not convert when I manually created the correct dimensions [79,95,69]
    imcalcInfo.output = [fileNames{badInds(f)}(1:end-4), '_resized.nii'];
    imcalcInfo.interp = config.interp;             % nearest neighbor, defined as number 5
    runImcalc_v2018(imcalcInfo); % runs the interpolation
    fileNames{badInds(f)} = imcalcInfo.output; % make this the output file 
end    

% investigate the resized.nii file
% a=MRIread('padded_brain_rand_resized.nii')
% unique(a.vol)

subj_T1_path = fileNames{end}; %forReg_ID231_T1_z69.nii
fileNames = fileNames(1:end-1); %resized.nii

%% Coregister all files to SPM's T1 %%
if config.doCoreg
    load Coreg_SPM12
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {SPM_T1_path}; % Reference is the SPM T1 file
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {subj_T1_path}; % source is the forReg_ID231_T1_z69.nii
    matlabbatch{1}.spm.spatial.coreg.estwrite.other = fileNames; % the brain_resized.nii file
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = config.interp;
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'SPM_';
    matlabbatch = spm_jobman('convert',matlabbatch);  % convert to current SPM version, uses the SPM convert function
    spm_jobman('run',matlabbatch);    

    for f = 1:nFiles
        filesepInds = strfind(fileNames{f}, filesep);
        if ~isempty(filesepInds) % goes into loop
            fileNames{f} = [fileNames{f}(1:filesepInds(end)), 'SPM_', fileNames{f}((filesepInds(end)+1):end)];
            % find the resized.nii file 
        else
            fileNames{f} = ['SPM_', fileNames{f}];
        end
    end
end
    
%% Coregister all files to Freesurfer's T1 %%
if config.doCoreg
    load Coreg_SPM12
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {FS_T1_path}; % reference FreesurferT1.nii file
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {SPM_T1_path}; % source SPM T1
    matlabbatch{1}.spm.spatial.coreg.estwrite.other = fileNames; % resized.nii
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = config.interp;
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'FS_';
    matlabbatch = spm_jobman('convert',matlabbatch);  % convert to current SPM version
    spm_jobman('run',matlabbatch);    

    for f = 1:nFiles
        filesepInds = strfind(fileNames{f}, filesep);
        if ~isempty(filesepInds) % goes into loop
            fileNames{f} = [fileNames{f}(1:filesepInds(end)), 'FS_', fileNames{f}((filesepInds(end)+1):end)];
        % FS_SPM_padded_brain_rand_resized.nii
        else
            fileNames{f} = ['FS_', fileNames{f}];
        end
    end
end

% investigate the FS_SPM_padded ... ni
% b=MRIread('FS_SPM_padded_brain_rand_resized.nii')
% unique(b.vol) % still 0

%% Project each file onto the cortical surface %%
for f = 1:nFiles
    surfaceFileName = [origFileNames{f}(1:(end-4)), '_surface.nii'];
        
    currCommand = ['mri_vol2surf --mov ', fileNames{f}, ' --o ', surfaceFileName, ...
        ' --hemi ', config.hemi, ' --regheader cvs_avg35_inMNI152', ...
        ' --', config.projectStr]; % config projectStr is 'projfrac-max 0 1 0.1'
    [status] = system([setupCommands, currCommand], '-echo')
    
    % creates the padded_brain_rand_surface.nii
    % c=MRIread('padded_brain_rand_surface.nii')
    % unique(c.vol) % still 0
    
    if freeview
        figName = ['PNGs/', origFileNames{f}(1:(end-4)), '_', config.hemi, '_az', num2str(config.azimuth), '.png'];
        freeviewCommand = [setupCommands, ' freeview -f ', freeviewSurfaceDir, config.surfaceFile, ...
            ':edgethickness=0:overlay=', surfaceFileName, ...
            ':overlay_threshold=', num2str(config.threshold(1)), ',', num2str(config.threshold(2)), ...
            ' --zoom 1.5 --cam azimuth ', num2str(config.azimuth), ' ', config.camStr, ...
            ' --screenshot ', figName];
        [status] = system(freeviewCommand)
    end
end