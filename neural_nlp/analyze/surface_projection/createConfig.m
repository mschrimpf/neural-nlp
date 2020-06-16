function config = createConfig(pass)
config.interp = 0;
config.projectStr = 'projfrac 1'%'projfrac 1' % 'projfrac-max 0 1 0.1'; %'projfrac 1' %'projfrac-max 0 1 0.1'; % 'projfrac 1'
config.threshold = [0.05 50];		% assuming performance is normalized between 0 and 1; if it�s between 0 and 100, use [0, 100]
config.surfaceFile = 'lh.inflated';
config.hemi = 'lh';
config.doCoreg = 1;
config.azimuth = 0;
if pass >= 2
    config.doCoreg = 1; % changed to 1
    config.azimuth = 180;
end
if pass >= 3
    config.surfaceFile = 'rh.inflated';
    config.hemi = 'rh';
end
if pass == 4
    config.azimuth = 0;
end
end

