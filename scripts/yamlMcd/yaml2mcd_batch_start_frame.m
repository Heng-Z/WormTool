%%
mcdf = Mcd_Frame;
% % folder = '/Users/hengzhang/Nutstore Files/worm-head-dynamics/data/PRC/MUS_CrimsonR/control/';
% save_folder = '/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/WormSim/PRC/MUS_CrimsonR/Control';
folder = '/Volumes/Lenovo/Pinjie/lpj paper videos 231/wen1109 exp 20210705';
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
    disp(filebname);
    mcd = mcdf.yaml2matlab(filebname,10);
    disp([matched_file{n}, ' ', num2str(mcd(1,1).FrameNumber)])
    
end


