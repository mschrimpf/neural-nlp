function dummy_run()
addpath("C:\Program Files\spm12");
dimensions = [88, 128, 85];
dummy_matrix = rand(dimensions);
volInfo = spm_vol('SPM_T1.nii');
filename = 'dummy.nii';
newVolInfo = struct('fname', filename, 'mat', volInfo.mat, 'dim',volInfo.dim, ...
    'dt',[spm_type('int16') spm_platform('bigend')],'pinfo',[1;0;0]);
spm_write_vol(newVolInfo, dummy_matrix);
projectOnSurfaceFigure([pwd, filename], createConfig(1));
end
