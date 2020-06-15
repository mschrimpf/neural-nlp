%subs = {'018', '199','215','288','289','296','343','366','407','426'}
subs = {'018','288','289','296','426'}

%% GPT2-xl
for g = subs
    b = join([pwd, '/brain_matrices/', g, '_Pereira2018_encoding_gpt2_xl.mat']);
    input = b{1};
    input = input(find(~isspace(input)));
    generateProjections(input, 'LH', 0) % for LH
    generateProjections(input, 'RH', 0) % for RH
end

%% Glove
for g = subs
    b = join([pwd, '/brain_matrices/', g, '_Pereira2018_encoding_glove.mat']);
    input = b{1};
    input = input(find(~isspace(input)));
    generateProjections(input, 'LH',0) % for LH
    generateProjections(input, 'RH',0) % for RH
end

%%
cd('/mindhive/evlab/u/gretatu/Desktop/surface_projection_v2/brain_matrices')

files = dir(fullfile(pwd, '*_Pereira2018_encoding_gpt2_xl_LH_surface.nii'))

%%
freeviewCommand = [' freeview -f $SUBJECTS_DIR/cvs_avg35_inMNI152/surf/lh.inflated', ...
    ':edgethickness=0:overlay=', files(10).name, ...
    ' --zoom 1.3 --cam azimuth 0'];
[status] = system(freeviewCommand)
