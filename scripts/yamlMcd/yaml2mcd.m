%%
mcdf = Mcd_Frame;
filebname = '/Volumes/Lenovo/Pinjie/lpj paper videos 205/N2/20190812/20190812_1429_w8.yaml';
mcd = mcdf.yaml2matlab(filebname,26000);

%%
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
%%
save("./20190812_1429_w8_centerline.mat","centerline","timestamp","stage_position","BoundaryA","BoundaryB")
