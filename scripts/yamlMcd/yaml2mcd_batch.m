%%
mcdf = Mcd_Frame;
% % folder = '/Users/hengzhang/Nutstore Files/worm-head-dynamics/data/PRC/MUS_CrimsonR/control/';
% save_folder = '/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/WormSim/PRC/MUS_CrimsonR/Control';
folder = '/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1109 exp 20210713/';
save_folder = '/Users/hengzhang/projects/WormSim/neuron_ablations/rawdata/PSi_processed/';
file_list = dir(folder);
matched_file = {};
for i = 1:length(file_list)
%Add the file name condition here
    if contains(file_list(i).name,'.yaml') && file_list(i).name(1) ~= '.'
        matched_file{end+1} = file_list(i).name;
    end
end

%%
for n = 1: length(matched_file)
    filebname = fullfile(folder,matched_file{n});
    save_name = fullfile(save_folder,[matched_file{n}(1:end-5),'.mat']);
    if exist(save_name,'file')
        disp('file exists, continue')
        continue;
    end
    disp(filebname);
    try
    mcd = mcdf.yaml2matlab(filebname,200000);
    catch
        disp('error in reading mcd file');
        continue;
    end
    centerline = zeros(100,2,size(mcd,2));
    stage_position = NaN(1,2,size(mcd,2));
    timestamp = zeros(size(mcd,2),1);
    BoundaryA = NaN(100,2,size(mcd,2));
    BoundaryB = NaN(100,2,size(mcd,2));
    for i = 1:size(mcd,2)
        centerline(:,1,i) = mcd(1,i).SegmentedCenterline(1:2:end);
        centerline(:,2,i) = mcd(1,i).SegmentedCenterline(2:2:end);
        timestamp(i) = mcd(1,i).TimeElapsed;
        try
        BoundaryA(:,1,i) = mcd(1,i).BoundaryA(1:2:end);
        BoundaryA(:,2,i) = mcd(1,i).BoundaryA(2:2:end);
        BoundaryB(:,1,i) = mcd(1,i).BoundaryB(1:2:end);
        BoundaryB(:,2,i) = mcd(1,i).BoundaryB(2:2:end);
        stage_position(1,1,i) = mcd(1,i).StagePosition(1);
        stage_position(1,2,i) = mcd(1,i).StagePosition(2);
        stage_position(1,1,i) = mcd(1,i).StagePosition(1);
        stage_position(1,2,i) = mcd(1,i).StagePosition(2);
        catch
        end
    end
    disp(save_name);
    save(save_name,'centerline','stage_position','timestamp','BoundaryA','BoundaryB');
end
%%
