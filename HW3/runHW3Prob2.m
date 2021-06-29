% This is the main script for Homework 3 Problem 1 for ECE5578.
% Clear the workspace.
clear all;
close all;

% Read the frames into memory.
im1 = double(rgb2gray(imread('foreman_yuv_150.png')));
im2 = double(rgb2gray(imread('foreman_yuv_151.png')));
im3 = double(rgb2gray(imread('foreman_yuv_152.png')));

% Display the original frames.
figure(1);
subplot(1,3,1)
imagesc(im1)
colormap('gray')
title('Foreman Frame #150')

subplot(1,3,2)
imagesc(im2)
colormap('gray')
title('Foreman Frame #151')

subplot(1,3,3)
imagesc(im3)
colormap('gray')
title('Foreman Frame #152')
close all;

% This will loop over the image, splitting it into blocks of size 8x8, and
% calling the function 'getBlkMotion' to estimate the block motion.
blkSize = 8;
[h,w,d] = size(im1);
xBlks = ceil(w/blkSize);
yBlks = ceil(h/blkSize);
imBlocks = cell(yBlks,xBlks);
for y = 1:yBlks
    yLoc = (y-1)*blkSize+1;
    if y == yBlks
        yRange = [yLoc:h];
    else
        yRange = [yLoc:y*blkSize];
    end
    for x = 1:xBlks
        xLoc = (x-1)*blkSize+1;
        if x == xBlks
            xRange = [xLoc:w];
        else
            xRange = [xLoc:x*blkSize];
        end
        imBlocks{y,x} = im2(yRange,xRange);
    end
end

% This loop will comput ethe motion vectors and the residual block. First,
% initialize the motion estimation range, as well as the pel value.
pel = 0.1;
if or(pel == 0.5,pel == 0.25)
    im1 = bilinearInterpolation(im1, (1/pel)*[h, w]);
    im2 = bilinearInterpolation(im2, (1/pel)*[h, w]);
    blkSize = blkSize*(1/pel);
end
range = 12;
mvBlks = cell(yBlks,xBlks);
resBlks = cell(yBlks,xBlks);
sadCounter = 0;
for y = 1:yBlks
    yLoc = (y-1)*blkSize+1;
    for x = 1:xBlks
        xLoc = (x-1)*blkSize+1;
        [mvBlks{y,x},resBlks{y,x},sadCount] = ...
            getDiamondSearchBlk(imBlocks{y,x},xLoc,yLoc,im1,pel,range);
        sadCounter = sadCounter + sadCount;
    end
end

% This loop will reconstruct the residual image from the residual blocks.
resImg = zeros(size(im1));
[h,w] = size(im1);
for y = 1:yBlks
    yLoc = (y-1)*blkSize+1;
    if y == yBlks
        yRange = [yLoc:h];
    else
        yRange = [yLoc:y*blkSize];
    end
    for x = 1:xBlks
        xLoc = (x-1)*blkSize+1;
        if x == xBlks
            xRange = [xLoc:w];
        else
            xRange = [xLoc:x*blkSize];
        end
        resImg(yRange,xRange) = resBlks{y,x};
        mvPlot.xLoc(y,x) = xLoc;
        mvPlot.yLoc(y,x) = yLoc;
        mvPlot.x(y,x) = mvBlks{y,x}.x;
        mvPlot.y(y,x) = mvBlks{y,x}.y;
    end
end

figure(2);
subplot(2,3,1)
imagesc(im1)
colormap('gray')
title('Frame #150')

subplot(2,3,4)
imagesc(im2)
colormap('gray')
title({'Original','Frame #151'})

subplot(2,3,2)
imagesc(double(abs(im2 - im1)))
colormap('gray')
title({'Actual','Residual'})

subplot(2,3,5)
imagesc(resImg)
colormap('gray')
title({'Predicted','Residual'})

subplot(2,3,3)
imagesc(im1 + double(abs(im2 - im1)))
colormap('gray')
title({'Actual','Reconstructed'})

subplot(2,3,6)
imagesc(im1 + resImg)
colormap('gray')
title({'Predicted','Reconstructed'})

figure(3)
imagesc(im1); hold on; grid on; colormap('gray');
quiver(mvPlot.xLoc(:),mvPlot.yLoc(:),mvPlot.x(:),mvPlot.y(:), '-r');
title('Motion Vectors');

% Compute the Mean Square Error of the Residual.
mse = 0;
for i = 1:h
    for j = 1:w
        mse = mse + power(resImg(i,j),2);
    end
end
sadCounter
mse = mse/(w*h)
return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% THE FOLLOWING FUNCTIONS ARE SUPPORT FUNCTIONS FOR HW3 %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the function provided by Dr. Li to estimate the block motion and
% the residual.
function [mv, res, sadCount]=getDiamondSearchBlk(blk, x0 , y0, ...
                                                 im, pel, range)
% INPUTS:
%  - blk - m x m blocks
%  - x0, y0 - blk location in current frame
%  - im - hxw a frame
%  - pel = 1, 0.5 0.25: pel steps
%  - range = 15; 
% output:
%   mv - motion vector
%   res - residual.

xStart = x0;
yStart = y0;

% Compute the block size.
[bh, bw] = size(blk);

% If the pel is a fraction, perform bilinear interpolation.
if pel == 0.5
    blk = bilinearInterpolation(blk, 2*[bh, bw]);
elseif pel == 0.25
    blk = bilinearInterpolation(blk, 4*[bh, bw]);
else
end
    
% Following Bilinear Interpolation, update the image and block size.
[h, w]=size(im);
[bh, bw]=size(blk);

% Initialize the motion vector.
mv.x=0;
mv.y=0;

% This loop will sweep over the range for each block to predict the motion
% vector. First, Initialize the parameters for computing the motion vector.
% Loop over the x range and the y range. First, Initialize all of the
% values.
reset = 1;
flag = 'l';
mad = inf*ones(2*range+1, 2*range+1);
mad_min=inf;
xMid = range+1;
yMid = range+1;
xMin = range+1;
yMin = range+1;
dXCent = range+1;
dYCent = range+1;
sadCount = 0;
progression = 0;
while 1
    
    % Reset the diamond Index.
    if reset
        diaIdx = initDiamond(flag);
        reset = 0;
    end
    
    for i = 1:size(diaIdx,1)
        % Update the center diamond ocation for the MAD and Motion Vector
        % variables.
        xLoc = dXCent + diaIdx(i,2);
        yLoc = dYCent + diaIdx(i,1);
        
        % Determine if the block-under-test is in the image and the block.
        inBlk = y0+diaIdx(i,1) > 0 && y0+diaIdx(i,1) <= (h-bh) && ...
            x0+diaIdx(i,2) > 0 && x0+diaIdx(i,2) <= (w-bw);
        inRng = yLoc > 0 && yLoc <= (2*range+1) && ...
            xLoc > 0 && xLoc <= (2*range+1);
        if ~inRng
            continue;
        elseif inBlk && inRng
            if mad(yLoc,xLoc) ~= inf
                continue;
            end
            % Capture the residual of the block.
            if (y0+diaIdx(i,1)+bh-1) <= h && (x0+diaIdx(i,2)+bw-1) <= w
                blk_res = double(abs(...
                    im(y0+diaIdx(i,1):y0+diaIdx(i,1)+bh-1, ...
                    x0+diaIdx(i,2): x0+diaIdx(i,2)+bw-1) - blk));
            else
                mad(yLoc,xLoc) = inf;
                sadCount = sadCount + 1;
%                 mvs(yLoc,xLoc).x = xLoc - xMid;
%                 mvs(yLoc,xLoc).y = yLoc - yMid;
                continue;
            end

            % Compute the mean absolute deviation of the residual block.
            mad(yLoc,xLoc) = mean(blk_res(:));
            sadCount = sadCount + 1;
            
            % If the mean absolute deviation of the residual block is the
            % minimum, save the motion vector as the 'x' and 'y' offsets,
            % and the residual block.
%             mvs(yLoc,xLoc).x = xLoc - xMid;
%             mvs(yLoc,xLoc).y = yLoc - yMid;
            if mad(yLoc,xLoc) < mad_min
                mad_min = mad(yLoc,xLoc);
%                 mv = mvs(yLoc,xLoc);
                mv.x = xLoc - xMid;%mvs(yLoc,xLoc);
                mv.y = yLoc - yMid;%mvs(yLoc,xLoc);
                res = blk_res;
                xMin = xLoc;
                yMin = yLoc;
            end
        else
            % If the mean absolute deviation of the residual block is not
            % in the block-under-test, move on.
            mad(yLoc,xLoc) = inf;
%             mvs(yLoc,xLoc).x = xLoc - xMid;
%             mvs(yLoc,xLoc).y = yLoc - yMid;
            sadCount = sadCount + 1;
        end
    end
    progression = progression + 1;
    if progression == 8
        fprintf('xStart = %d, yStart = %d\n',xStart,yStart)
    end
    % Make adjustments to the diamond based on the results of the MADs.
    if strcmp(flag,'s')
        break;
    elseif xMin == dXCent && yMin == dYCent
        flag = 's';
        reset = 1;
    else
        dXCent = xMin;
        dYCent = yMin;
        x0 = x0 + dXCent - xMid;
        y0 = y0 + dYCent - yMid;
    end
end
end

% This function will receive an input image and will perform bilinear
% interpolation to generate a new output image with a new set of
% dimensions.
function [out] = bilinearInterpolation(im, out_dims)
% INPUTS:
%  - im - hxw a frame
%  - out_dims - the desired output dimensions
% output:
%  - out - the bilinearly interpolated image.

% Initialize the desired output rows from the size of the input image.
in_rows = size(im,1);
in_cols = size(im,2);
out_rows = out_dims(1);
out_cols = out_dims(2);

% Determine the scale of the rows and columns.
rowScale = in_rows / out_rows;
colScale = in_cols / out_cols;

% Define grid of coordinates in the new image.
[cf, rf] = meshgrid(1 : out_cols, 1 : out_rows);

% Compute the row range and column range of the original image.
rf = rf * rowScale;
cf = cf * colScale;
r = floor(rf);
c = floor(cf);

% Clip any values out of the appropriate range.
r(r < 1) = 1;
c(c < 1) = 1;
r(r > in_rows - 1) = in_rows - 1;
c(c > in_cols - 1) = in_cols - 1;

% Establish the stepsize.
delta_R = rf - r;
delta_C = cf - c;

% Generate the new indices of the new rows and columns.
in1_ind = sub2ind([in_rows, in_cols], r, c);
in2_ind = sub2ind([in_rows, in_cols], r+1,c);
in3_ind = sub2ind([in_rows, in_cols], r, c+1);
in4_ind = sub2ind([in_rows, in_cols], r+1, c+1);       

% Interpolate the new values along the new indices.
out = zeros(out_rows, out_cols, size(im, 3));
out = cast(out, class(im));
for idx = 1 : size(im, 3)
    chan = double(im(:,:,idx));
    tmp = chan(in1_ind).*(1 - delta_R).*(1 - delta_C) + ...
        chan(in2_ind).*(delta_R).*(1 - delta_C) + ...
        chan(in3_ind).*(1 - delta_R).*(delta_C) + ...
        chan(in4_ind).*(delta_R).*(delta_C);
    out(:,:,idx) = cast(tmp, class(im));
end
end

% This function will initialize the large diamond for performing large
% diamond searching on motion vectors.
function [diaIdx] = initDiamond(sw)
% INPUTS:
%  - sw - input switch.
%          - Case = 'l': initialize the large diamond.
%          - Case = 's': initialize the small diamond.
% OUTPUTS:
%  - diaIdx - a matrix of the upper left corner of the blocks to compare.

% This is the main switch. If
if strcmp(sw,'l')
    diaIdx = zeros(9,2);
    diaIdx(1,:) = [0, 0];
    diaIdx(2,:) = [0, 2];
    diaIdx(3,:) = [1, 1];
    diaIdx(4,:) = [2, 0];
    diaIdx(5,:) = [1, -1];
    diaIdx(6,:) = [0, -2];
    diaIdx(7,:) = [-1, -1];
    diaIdx(8,:) = [-2, 0];
    diaIdx(9,:) = [-1, 1];
elseif strcmp(sw,'s')
    diaIdx = zeros(5,2);
    diaIdx(1,:) = [0, 0];
    diaIdx(2,:) = [0, 1];
    diaIdx(3,:) = [1, 0];
    diaIdx(4,:) = [0, -1];
    diaIdx(5,:) = [-1, 0];
end
end