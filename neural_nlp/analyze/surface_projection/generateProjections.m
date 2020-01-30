function generateProjections(matfile, inv)
% matfile is the 3D matrix with ROI or brainscore coordinates. Contains
% info about the subject ID and model

addpath('/software/spm12');
volInfo = spm_vol('SPM_T1.nii');
filename = strcat(matfile(1:end-3), 'nii');

% Load the brain matrix
cd('/mindhive/evlab/u/gretatu/Desktop/surface_projection/');
brain = load(matfile);
brain_matrix = brain.brain_matrix;
brain_matrix = brain_matrix * 100;

% For 3d plotting
% [X,Y,Z] = ndgrid(1:size(brain_matrix,1), 1:size(brain_matrix,2), 1:size(brain_matrix,3));
% pointsize = 30;
% figure;scatter3(X(:), Y(:), Z(:), pointsize, brain_matrix(:));

% Write to volume, .nii
newVolInfo = struct('fname', filename, 'mat', volInfo.mat, 'dim',size(brain_matrix), ...
    'dt',[spm_type('int16') spm_platform('bigend')],'pinfo',[1;0;0]);

spm_write_vol(newVolInfo, brain_matrix); 

if inv == 1
    % Count number of non-nans
    fprintf('Non-nan values in brain_matrix: %.f \n and mean: \n', nnz(~isnan(brain_matrix)), nanmean(nanmean(nanmean(brain_matrix))))
    
    % In the .nii file
    vol_output = MRIread(filename) 
    fprintf('Non-zero values in volume brain_matrix (after spm_write_vol): %.f \n Dimensions: %.f %.f %.f \n', nnz((vol_output.vol ~= 0)), [size(vol_output.vol)])
    
    % In the file that is coregistered to SPM's T1
    spm_output = MRIread(strcat('SPM_', filename))
    fprintf('Non-zero values in SPM brain_matrix (after SPM coreg): %.f \n Dimensions: %.f %.f %.f \n', nnz((spm_output.vol ~= 0)), [size(spm_output.vol)])
    
     % In the file that is coregistered to Freesurfer's T1
    fs_spm_output = MRIread(strcat('FS_SPM_', filename))
    fprintf('Non-zero values in Freesurfers T1 brain_matrix: %.f \n Dimensions: %.f %.f %.f \n', nnz((fs_spm_output.vol ~= 0)), [size(fs_spm_output.vol)])
    
     % In the surface projected file 
     surf_output = MRIread(strcat(filename(1:(end-4)), '_surface.nii'))
    fprintf('Non-zero values in surface projected file: %.f \n Dimensions: %.f %.f %.f \n', nnz((surf_output.vol ~= 0)), [size(surf_output.vol)])
    
    % u=unique(brain_matrix);

end     

% RUN PLOTTING CODE
projectOnSurfaceFigure_greta({[pwd, '/', filename]}, createConfig(1), 0);

end

% for freesurfer
% export FREESURFER_HOME=/software/Freesurfer/5.3.0; export SUBJECTS_DIR=/software/Freesurfer/5.3.0/subjects; source $FREESURFER_HOME/SetUpFreeSurfer.sh;  freeview -f /software/Freesurfer/5.3.0/subjects/cvs_avg35_inMNI152/surf/lh.inflated:edgethickness=0:overlay=/mindhive/evlab/u/gretatu/Desktop/surface_projection/braintest4_surface.nii:overlay_threshold=0.05,0.4 

% rh, azimuth 180
% export FREESURFER_HOME=/software/Freesurfer/5.3.0; export SUBJECTS_DIR=/software/Freesurfer/5.3.0/subjects; source $FREESURFER_HOME/SetUpFreeSurfer.sh;  freeview -f /software/Freesurfer/5.3.0/subjects/cvs_avg35_inMNI152/surf/rh.inflated:edgethickness=0:overlay=/mindhive/evlab/u/gretatu/Desktop/surface_projection/Pereira2018-encoding-glove_surface.nii:overlay_threshold=0.05,50 --cam azimuth 180

