function mcdf=yaml2matlab(file,endFrame)
% This function reads in a yaml file produced by the MindControl Software
% and exports an array of MindControl Data Frames (mcdf's) that is easy to
% manipulate in matlab.
%
% Andrew Leifer
% leifer@fas.harvard.edu
% 2 November 2010


fid = fopen(file); 

Mcd_Frame.seekToFirstFrame(fid);
k=1;
error_count = 0;
while(~feof(fid))
    try 
    mcdf(k)=Mcd_Frame.readOneFrame(fid); %#ok<AGROW>
    catch
        disp('error reading frame')
        break;
%     mcdf(k)= mcdf(k-1);
%     error_count = error_count + 1;
    end
    k=k+1;
    if ~mod(k,100)
        disp(k);
        if error_count >0
           disp(['error count: ' num2str(error_count)]);
           error_count = 0;
        end

    end
    if k > endFrame % DIY
        break;
    end
end
fclose(fid);

