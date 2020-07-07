function runImcalc_v2018(data)

% Run SPM's imcalc function
%
% INPUT:
% data = structure with the following fields:
%        input = cell of strings, full paths to files
%                (must be all .nii or all .img)
%        output = string, full path to output file
%        expression = string, expression to calculate
%                     (e.g., 'i1+2*i2')
%        interp (optional) = 1 for trilinear (default), 0 for nearest-neighbor

load imcalcBatch_SPM5   % using SPM5 version because SPM12 does not appear to resize images when their dimensions do not match
jobs{1}.util{1}.imcalc.input = data.input;
jobs{1}.util{1}.imcalc.output = data.output;
jobs{1}.util{1}.imcalc.outdir = [];            
jobs{1}.util{1}.imcalc.expression = data.expression;
if isfield(data, 'interp')
    jobs{1}.util{1}.imcalc.options.interp = data.interp;    
end
   
theVersion = which('spm');
if isempty(strfind(theVersion, 'spm5'))
    jobs = spm_jobman('convert',jobs);  % convert to current SPM version
end
spm_jobman('run',jobs);    