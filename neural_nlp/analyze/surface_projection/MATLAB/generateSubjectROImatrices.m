% Generates matrices containing subject-specific ROIs for ROIs across all brain networks,
% or only for a particular network's ROIs.
% Counts overlapping voxels among networks and network statistics.

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
   
    % indices 4, 12, 20, 24, 28 for 90% extraction
    col_idx_store = [];

    % loop across lang, MD, DMN, auditory, visual 
    for j = [4, 12, 20, 24, 28];
    % for j = [12];
        
        % count number of voxels for each network
        network_count = 0;
        
        for i = 1:length(meta.roiColumns{j, 1});
            
            col_idx = meta.roiColumns{j, 1}{i, 1}; % i=1, first ROI - column indices

            for k = 1:length(col_idx) % for each voxel

                % store all unique column indices across ROIs
                col_idx_store = [col_idx_store; col_idx(k)];

                idx = meta.colToCoord(col_idx(k),:,:); % col_idx(k) contains all the unique column indices
 
                ROI_matrix(idx(1),idx(2),idx(3)) = 1; % assign binary value
            end    
            
            network_count = network_count + length(col_idx);

        end
        
        % count the number of voxels per network
        network_name = meta.atlases{j, 1};
        subjects_voxels{count}.(network_name) = network_count;
        
        display(network_count)

        % extract matrix for only language ROIs (all 12)
        if j == 4
            save(['ROIs/', filename(1:3), '_langROIs.mat'], 'ROI_matrix');
            display(nnz(~isnan(ROI_matrix)));
        end
        
    end
   
    % save matrix with all ROI coordinates
    save(['ROIs/', filename(1:3), '_allROIs.mat'], 'ROI_matrix');
    
    % count number of unique voxels
    unique_voxels = length(unique(col_idx_store)); % or nnz(~isnan(ROI_matrix))
    
    % count overlapping voxels
    overlap = length(col_idx_store) - unique_voxels;
    
    if length(col_idx_store) ~= total_voxels
        error('Discrepancy between total number of voxels for subject: %s', filename)
    end
    
    subjects_ROIs.matfile{1,count} = filename;
    subjects_ROIs.ROI_overlap{1,count} = overlap;
    subjects_ROIs.total_unique_voxels_90{1,count} = unique_voxels;
    subjects_ROIs.total_voxels_90{1,count} = length(col_idx_store);
    
    end
end


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
