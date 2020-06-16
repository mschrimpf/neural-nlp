function generateProjections(brainfile, hemi, language)
% e.g. generateProjections([pwd, '/brain_matrices/018_langROIs.mat'], 'LH')
% 
% INPUT
%
% brainfile = string, path to the 3D matrix with 3D matrix with brainscore coordinates.
% hemi = string, either 'LH' or 'RH'.

% PATHS
addpath('/cm/shared/openmind/freesurfer/6.0.0/matlab/');
addpath('/om/group/evlab/software/spm12');

% ADDITIONAL PATHS - load the EvLab17 pipeline and freesurfer
% addpath('/om/group/evlab/software/evlab17');
% evlab17 init
% addpath('/cm/shared/openmind/xjview/xjview97/xjview');
% spm fmri

% TESTING
% brainfile='/mindhive/evlab/u/gretatu/Desktop/surface_projection_v2/brain_matrices/407_Pereira2018_encoding_glove.mat'
% configNum = 1;
% hemi = 'LH';

if hemi == 'LH'
    configNum = 1;
else hemi == 'RH'
    configNum = 3;
end

subj_T1_path = fullfile(pwd,'forReg_ID231_T1_z69.nii');
volInfo = spm_vol(subj_T1_path) 

fb = strcat(brainfile(1:end-4), '_', hemi, '.nii');

% Load the files
brain = load(brainfile);
brain_matrix = brain.brain_matrix; 

% Set negative and zero values to 0.00001
brain_matrix(brain_matrix < 0) = 0.01;

% min(min(min(brain_matrix)))


% If using abs values
% brain_matrix = (abs(brain_matrix));

% Plot histogram of values
% brain_hist = squeeze(reshape(brain_matrix,[1 79*95*69]));
% figure;histogram(brain_hist(brain_hist > 0.001), 500);title('Raw values, neg removed');ylim([0 90])

% figure;histogram(brain_hist, 50);

% For 3d plotting
% [X,Y,Z] = ndgrid(1:size(brain_matrix,1), 1:size(brain_matrix,2), 1:size(brain_matrix,3));
% pointsize = 30;
% figure;scatter3(X(:), Y(:), Z(:), pointsize, brain_matrix(:));

% Write to volume, .nii - using the volinfo dimensions
volInfo_brain = struct('fname', fb, 'mat', volInfo.mat, 'dim',volInfo.dim, ...
    'dt',[spm_type('float32') spm_platform('bigend')],'pinfo',[1;0;0]);

spm_write_vol(volInfo_brain, brain_matrix); 

% RUN PLOTTING CODE
transformToSurface({fb}, createConfig(configNum), 1);

% Constrain ROIs

if language
brain_surf = [fb(1:end-4), '_surface.nii']

nameIdx = strfind(fb, '/'); % idx 
subjectID = fb(nameIdx(end)+1:nameIdx(end)+3)

ROI_surf = [pwd, '/ROIs/', subjectID, '_langROIs_', hemi, '_surface.nii']

brain_surf_o = MRIread(brain_surf);
ROI_surf_o = MRIread(ROI_surf);

brain_surf_c = brain_surf_o; % copy, constrain
brain_surf_c.vol(ROI_surf_o.vol == 0) = 0;
     
MRIwrite(brain_surf_c, [fb(1:end-4), '_surface_lang.nii'])

end
end