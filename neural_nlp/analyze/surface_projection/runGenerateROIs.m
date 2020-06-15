subs = {'018', '199','215','288','289','296','343','366','407','426'}

%% MD
% for g = subs
%     b = join([pwd, '/ROIs/',g,'_MDROIs.mat']);
%     input = b{1};
%     input = input(find(~isspace(input)));
%     generateROIs(input, 1) % for LH
%     generateROIs(input, 3) % for RH
% end

%% DMN
for g = subs
    b = join([pwd, '/ROIs/',g,'_DMNROIs.mat']);
    input = b{1};
    input = input(find(~isspace(input)));
    generateROIs(input, 1) % for LH
    generateROIs(input, 3) % for RH
end

%% Auditory
for g = subs
    b = join([pwd, '/ROIs/',g,'_auditoryROIs.mat']);
    input = b{1};
    input = input(find(~isspace(input)));
    generateROIs(input, 1) % for LH
    generateROIs(input, 3) % for RH
end

%% Visual
for g = subs
    b = join([pwd, '/ROIs/',g,'_visualROIs.mat']);
    input = b{1};
    input = input(find(~isspace(input)));
    generateROIs(input, 1) % for LH
    generateROIs(input, 3) % for RH
end

%% Language
for g = subs
    b = join([pwd, '/ROIs/',g,'_langROIs.mat']);
    input = b{1};
    input = input(find(~isspace(input)));
    generateROIs(input, 1) % for LH
    generateROIs(input, 3) % for RH
end

%% All ROIs (mask)
for g = subs
    b = join([pwd, '/ROIs/',g,'_allROIs.mat']);
    input = b{1};
    input = input(find(~isspace(input)));
    generateROIs(input, 1) % for LH
    generateROIs(input, 3) % for RH
end
    