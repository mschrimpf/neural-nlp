function generateROIs(ROIfile, configNum)
% Example: generateROIs([pwd, '/ROIs_v1/199_langROIs.mat'], 1)
% Separate script for generating labels in Freesurfer (createLabel.sh)

% PATHS
addpath('/cm/shared/openmind/freesurfer/6.0.0/matlab/');
addpath('/om/group/evlab/software/spm12');

% TESTING
% ROIfile='/mindhive/evlab/u/gretatu/Desktop/surface_projection/ROIs_v1/018_langROIs.mat'

subj_T1_path = fullfile(pwd,'forReg_ID231_T1_z69.nii');
volInfo = spm_vol(subj_T1_path) 

if configNum == 1
    filename_ROI = strcat(ROIfile(1:end-4), '_LH.nii');
else configNum == 3
    filename_ROI = strcat(ROIfile(1:end-4), '_RH.nii');
end

% Load the files
ROI = load(ROIfile);
ROI_matrix = ROI.ROI_matrix; 

volInfo_ROI = struct('fname', filename_ROI, 'mat', volInfo.mat, 'dim',volInfo.dim, ...
    'dt',[spm_type('int16') spm_platform('bigend')],'pinfo',[1;0;0]);

spm_write_vol(volInfo_ROI, ROI_matrix); 

% RUN PLOTTING CODE
transformToSurface({filename_ROI}, createConfig(configNum), 0);

end
