%%
mcdf = Mcd_Frame;
% folder_ls = ["/Volumes/Lenovo/Neck_Inhibition/full"];
folder_ls = ["/Users/hengzhang/Downloads/RMD_inhibition/"];
matched_file = {};
save_folder = "/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/WormSim/neuron_ablations/rawdata/RMD_inhibition/";
for f= 1: length(folder_ls)
    file_list = dir(folder_ls(f));
    for i = 1:length(file_list)
    %Add the file name condition here
        if contains(file_list(i).name,'.mat') && file_list(i).name(1) ~= '.'
            matched_file{end+1} = fullfile(folder_ls(f),file_list(i).name);
        end
    end
end

%%
for n = 1:length(matched_file)
    load(matched_file{n});
    disp(['loading ', char(matched_file{n})])
    centerline = zeros(100,2,size(mcd,2));
    stage_position = NaN(1,2,size(mcd,2));
    timestamp = zeros(size(mcd,2),1);
    BoundaryA = NaN(100,2,size(mcd,2));
    BoundaryB = NaN(100,2,size(mcd,2));
    frame_number = zeros(size(mcd,2),1);
    IllumSide = zeros(size(mcd,2),1);
    DLP = zeros(size(mcd,2),1);
    for i = 1:size(mcd,2)
        centerline(:,1,i) = mcd(1,i).SegmentedCenterline(1:2:end);
        centerline(:,2,i) = mcd(1,i).SegmentedCenterline(2:2:end);
        timestamp(i) = mcd(1,i).TimeElapsed;
        frame_number(i) = mcd(1,i).FrameNumber;
        try
        BoundaryA(:,1,i) = mcd(1,i).BoundaryA(1:2:end);
        BoundaryA(:,2,i) = mcd(1,i).BoundaryA(2:2:end);
        BoundaryB(:,1,i) = mcd(1,i).BoundaryB(1:2:end);
        BoundaryB(:,2,i) = mcd(1,i).BoundaryB(2:2:end);
        stage_position(1,1,i) = mcd(1,i).StagePosition(1);
        stage_position(1,2,i) = mcd(1,i).StagePosition(2);
        stage_position(1,1,i) = mcd(1,i).StagePosition(1);
        stage_position(1,2,i) = mcd(1,i).StagePosition(2);
        if mcd(1,i).DLPisOn == 1
            IllumSide(i) = mcd(1,i).IllumRectOrigin(1);
        end
        DLP(i) = mcd(1,i).DLPisOn;
        catch
        end
    end
    name_char = char(matched_file{n});
    % split the name by '/'
    root_name = split(name_char,'/');
    % split the name by '.'
    root_name = split(root_name(end),'.');
    save_name = fullfile(save_folder,root_name(1)+".mat");
    save(save_name,"centerline","timestamp","stage_position","BoundaryA","BoundaryB","IllumSide","frame_number","DLP");
end
