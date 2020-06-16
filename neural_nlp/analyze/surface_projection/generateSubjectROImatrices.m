% Generates matrices containing subject-specific ROIs for ROIs across all brain networks,
% or only for a particular network's ROIs.
% Counts overlapping voxels among networks. Counts ROI and network sizes.

files = dir(fullfile(pwd, '*persent.mat')); 
clear subjects_ROIs
clear subjects_voxels
count = 0;
filename_prev = 'mock' % only loop over unique subjects

for g = 1:length(files)
    
    if g == 1
        filename = files(g).name;
    else
        filename_prev = filename;
        filename = files(g).name;
    end    
    
    filename = files(g).name;
    subj = load(filename);
    
    if filename(1:3) == filename_prev(1:3)
        continue
    else
    count = count + 1; % if running all subjects, both sessions, remove the count and loop, and index by g
       
    meta = subj.meta;

    if meta.dimensions ~= [79,95,69]
        error('SPM dimensions inconsistent for subject file: %s', filename)
    else
        ROI_matrix = NaN(meta.dimensions);
    end 
   
    col_idx_store = []; % store all unique column indices across ROIs 
    % (i.e. it is possible to check when networks have overlapping column indices)
    ROI_size = []; % store size of each ROI

    % loop across lang, MD, DMN, auditory, visual 
    % indices 4, 12, 20, 24, 28 for 90% extraction
    for j = [4, 12, 20, 24, 28];
    % for j = [12];
        
        % count number of voxels for each network
        network_count = 0;
        network_unique = []; % find unique indices per network to compare which networks overlap
        
        for i = 1:length(meta.roiColumns{j, 1});
            
            col_idx = meta.roiColumns{j, 1}{i, 1}; % i=1, first ROI - column indices

            for k = 1:length(col_idx) % for each voxel

                col_idx_store = [col_idx_store; col_idx(k)];

                idx = meta.colToCoord(col_idx(k),:,:); % col_idx(k) contains all the unique column indices
 
                ROI_matrix(idx(1),idx(2),idx(3)) = 1; % assign binary value
                
                network_unique = [network_unique; col_idx(k)];
            end    

            ROI_size = [ROI_size; length(col_idx)]; % append size of each single ROI
            network_count = network_count + length(col_idx);

        end
        
        % count the number of voxels per network
        network_name = meta.atlases{j, 1};
        subjects_voxels{count}.(network_name) = network_count;
        subjects_voxels{count}.uniqueVoxels.(network_name) = network_unique;

        % extract matrix for only language ROIs (all 12)
        if j == 4
            % save(['ROIs/', filename(1:3), '_langROIs.mat'], 'ROI_matrix');
            display(nnz(~isnan(ROI_matrix)));
        end
        
    end
   
    % save matrix with all ROI coordinates
    % save(['ROIs/', filename(1:3), '_allROIs.mat'], 'ROI_matrix');
    
    % count number of unique voxels
    unique_voxels = length(unique(col_idx_store)); % or nnz(~isnan(ROI_matrix))
    % for each network, save unique col indices and compare across these
    % later 
    
    % count overlapping voxels
    overlap = length(col_idx_store) - unique_voxels;
    
    subjects_ROIs.matfile{1,count} = filename;
    subjects_ROIs.ROI_overlap{1,count} = overlap;
    subjects_ROIs.total_unique_voxels_90{1,count} = unique_voxels;
    subjects_ROIs.total_voxels_90{1,count} = length(col_idx_store);
    subjects_ROIs.ROI_size_90{1,count} = ROI_size;
    
    end
end

% save('subjects_ROIs_2020_04_02.mat','subjects_ROIs')
% save('subjects_voxels_2020_04_02.mat','subjects_voxels')

%% check intersection among networks
lang_u = subjects_voxels{1,1}.uniqueVoxels.language_from90to100prcnt;
md_u = subjects_voxels{1,1}.uniqueVoxels.MD_HminusE_from90to100prcnt;
dmn_u = subjects_voxels{1, 1}.uniqueVoxels.DMN_FIXminusH_from90to100prcnt;
aud_u = subjects_voxels{1, 1}.uniqueVoxels.auditory_from90to100prcnt;
vis_u = subjects_voxels{1, 1}.uniqueVoxels.visual_from90to100prcnt;

sum_all = sum([length(lang_u), length(md_u), length(dmn_u), length(aud_u), length(vis_u)]) % corresponds to total_voxels
unique_vox = unique(vertcat((lang_u), (md_u), (dmn_u), (aud_u), (vis_u)))

sum_all - length(unique_vox)

intersect(lang_u, md_u)
intersect(lang_u, dmn_u)
intersect(lang_u, aud_u)
intersect(lang_u, vis_u)

intersect(md_u, dmn_u)
intersect(md_u, aud_u)
intersect(md_u, vis_u)

intersect(dmn_u, aud_u)
intersect(dmn_u, vis_u)

intersect(aud_u, vis_u)

sum([length(intersect(lang_u, md_u)), length(intersect(lang_u, dmn_u)), length(intersect(lang_u, aud_u)), ...
length(intersect(lang_u, vis_u)), length(intersect(md_u, dmn_u)), length(intersect(md_u, aud_u)), length(intersect(md_u, vis_u)), ...
length(intersect(dmn_u, aud_u)), length(intersect(dmn_u, vis_u)), length(intersect(aud_u, vis_u))])

%% iterate over the ROI sizes and compare with the extracted ROI sizes for xarray

subject_IDs = {'018', '199', '215', '288', '289', '296', '343', '366', '407','426'};

diff_store = [];
diff_persub = zeros(10,54)

for subj = 1:length(subjects_ROIs.ROI_size_90)

    roi_sizes = subjects_ROIs.ROI_size_90{1, subj};
    fname = strcat('noAnnot_', subject_IDs(subj), '_ROI_sizes_2020-04-02_gpt2-xl.csv')
    roi_sizes_xarray = load(char(fname));
    
    test = isequal(roi_sizes, roi_sizes_xarray)
    
    diff = roi_sizes - roi_sizes_xarray;
    diff_store = [diff_store; diff];
    
    diff_persub(subj, :) = diff;
end

figure;
title('Discrepancy between xarray ROI sizes and MATLAB ROI sizes')
hold on
scatter([1:540], diff_store, 3, 'filled')
hold on
for val = [54:54:540]
    xline(val)
end

%% Manually sum over the network values for 215
length(cell2mat(sub215.meta.roiColumns{4,:})) + length(cell2mat(sub215.meta.roiColumns{12,:})) + length(cell2mat(sub215.meta.roiColumns{20,:})) + length(cell2mat(sub215.meta.roiColumns{24,:})) + length(cell2mat(sub215.meta.roiColumns{28,:}))



%% network statistics
lang = [];
md = [];
dmn = [];
aud = [];
vis = [];

for i = 1:10
lang = [lang; subjects_voxels{1, i}.language_from90to100prcnt];
md = [md; subjects_voxels{1, i}.MD_HminusE_from90to100prcnt];
dmn = [dmn; subjects_voxels{1, i}.DMN_FIXminusH_from90to100prcnt];
aud = [aud; subjects_voxels{1, i}.auditory_from90to100prcnt];
vis = [vis; subjects_voxels{1, i}.visual_from90to100prcnt];
end 

network_means = mean([lang, md, dmn, aud, vis])
network_stds = std([lang, md, dmn, aud, vis])
