mcdf = Mcd_Frame;
% folder_ls = ["/Volumes/Lenovo/Neck_Inhibition/full"];
% folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20200928","/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20200929", ...
%     "/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20201024","/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20201025"];
% folder_ls = ["/Volumes/Lenovo/Pinjie/wen1037_flp22__minisog/video/"];
% folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1119 exp 20210805/", "/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1119 exp 20210812/"];
% folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1123 exp 20211117", "/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1123exp 20211126"];
folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 205/wen1111 rmdsmd_minisog_agar/20210206", "/Volumes/Lenovo/Pinjie/lpj paper videos 205/wen1111 rmdsmd_minisog_agar/20210310/exp"]
matched_file = {};
save_folder = 'SMDSMBk_processed';
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
        
        catch
        end
    end
    name_char = char(matched_file{n});
    % split the name by '/'
    root_name = split(name_char,'/');
    % split the name by '.'
    root_name = split(root_name(end),'.');
    save_name = fullfile(save_folder,root_name(1)+".mat");
%     save(save_name,"centerline","timestamp","stage_position","BoundaryA","BoundaryB","IllumSide","frame_number","DLP");
    disp(['saving ', char(save_name)])
    save(save_name,"centerline","timestamp","stage_position","BoundaryA","BoundaryB")
end