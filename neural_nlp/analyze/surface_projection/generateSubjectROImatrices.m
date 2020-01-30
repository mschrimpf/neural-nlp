% Generates matrices containing subject-specific ROIs for ROIs across all
% brain networks, or only language ROIs

files = dir(fullfile(pwd, '*persent.mat')); 
clear subjects_ROIs
clear subjects_voxels
count = 0;

% only loop over unique subjects
for g = 1:length(files)

    filename_prev = filename;
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
   
    % Indices 4, 12, 20, 24, 28 for 90%
    col_idx_store = [];

    % Loop across lang, MD, DMN, auditory, visual 
    for j = [4, 12, 20, 24, 28];
        
        % count number of voxels for each network
        network_count = 0;
        
        for i = 1:length(meta.roiColumns{j, 1});
            
            col_idx = meta.roiColumns{j, 1}{i, 1}; % i=1, first lang region - column indices

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
    
    subjects_ROIs.matfile{1,count} = filename;
    subjects_ROIs.ROI_overlap{1,count} = overlap;
    subjects_ROIs.total_unique_voxels_90{1,count} = unique_voxels;
    subjects_ROIs.total_voxels_90{1,count} = length(col_idx_store);
    
    end
end





