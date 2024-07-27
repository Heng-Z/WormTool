%%
mcdf = Mcd_Frame;
% folder_ls = ["/Volumes/Lenovo/Neck_Inhibition/full"];
% folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20200928","/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20200929", ...
%     "/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20201024","/Volumes/Lenovo/Pinjie/lpj paper videos 51/wen1101/wen1101 exp 20201025"];
% folder_ls = ["/Volumes/Lenovo/Pinjie/wen1037_flp22__minisog/video/"];
% folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1119 exp 20210805/", "/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1119 exp 20210812/"];
% folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 205/wen1111 rmdsmd_minisog_agar/20210206", "/Volumes/Lenovo/Pinjie/lpj paper videos 205/wen1111 rmdsmd_minisog_agar/20210310/exp"]
% folder_ls = ["/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1123 exp 20211117", "/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1123exp 20211126"];
folder_ls = ["/Users/hengzhang/Downloads/RMD_inhibition"];

matched_file = {};
for f= 1: length(folder_ls)
    file_list = dir(folder_ls(f));
    for i = 1:length(file_list)
    %Add the file name condition here
        if contains(file_list(i).name,'.yaml') && file_list(i).name(1) ~= '.'
            matched_file{end+1} = fullfile(folder_ls(f),file_list(i).name);
        end
    end
end

%%
for n = 1: length(matched_file)
    filebname = char(matched_file{n});
    save_name = [filebname(1:end-5),'.mat'];
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
    
    save(save_name,'mcd');
end
%%
