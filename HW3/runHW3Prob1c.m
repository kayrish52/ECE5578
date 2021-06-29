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
blkSize = 4;
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
pel = 1;
if or(pel == 0.5,pel == 0.25)
    im1 = bilinearInterpolation(im1, (1/pel)*[h, w]);
    im2 = bilinearInterpolation(im2, (1/pel)*[h, w]);
    im3 = bilinearInterpolation(im3, (1/pel)*[h, w]);
end
range = 12;
mv1Blks = cell(yBlks,xBlks);
mv2Blks = cell(yBlks,xBlks);
resBlks = cell(yBlks,xBlks);
sadCounter = 0;
for y = 1:yBlks
    yLoc = (y-1)*blkSize+1;
    for x = 1:xBlks
        xLoc = (x-1)*blkSize+1;
        [mv1Blks{y,x},mv2Blks{y,x},resBlks{y,x},sadCount] = ...
            getBiDirectionalMotion(imBlocks{y,x},xLoc,yLoc,...
                                   im1,im3,pel,range);
        sadCounter = sadCounter + sadCount;
    end
end

% This loop will reconstruct the residual image from the residual blocks.
resImg = zeros(size(im1));
if or(pel == 0.5,pel == 0.25)
    blkSize = blkSize*(1/pel);
end
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
        mv1Plot.xLoc(y,x) = xLoc;
        mv1Plot.yLoc(y,x) = yLoc;
        mv1Plot.x(y,x) = mv1Blks{y,x}.x;
        mv1Plot.y(y,x) = mv1Blks{y,x}.y;
        mv2Plot.xLoc(y,x) = xLoc;
        mv2Plot.yLoc(y,x) = yLoc;
        mv2Plot.x(y,x) = mv2Blks{y,x}.x;
        mv2Plot.y(y,x) = mv2Blks{y,x}.y;
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
quiver(mv1Plot.xLoc(:),mv1Plot.yLoc(:),mv1Plot.x(:),mv1Plot.y(:), '-r');
quiver(mv2Plot.xLoc(:),mv2Plot.yLoc(:),mv2Plot.x(:),mv2Plot.y(:), '-g');
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
% This function uses two separate images to predict the center image.
function [mv0, mv1, res, sadCount] = getBiDirectionalMotion(blk,x0,y0,...
                                                            im0,im1,...
                                                            pel, range)
% INPUTS:
%  - blk - m x m blocks
%  - x0, y0 - blk location in current frame
%  - im - hxw a frame
%  - pel = 1, 0.5 0.25: pel steps
%  - range = 8; 
% OUTPUTS:
%  - mv - motion vector
%  - res - residual.

% Compute the block size.
[bh, bw] = size(blk);

% If the pel is a fraction, perform bilinear interpolation.
if pel == 0.5
    blk = bilinearInterpolation(blk, 2*[bh, bw]);
    x0=x0*2; y0=y0*2; 
elseif pel == 0.25
    blk = bilinearInterpolation(blk, 4*[bh, bw]);
    x0=x0*4; y0=y0*4;
else
end
    
% This loop will sweep over the range for each block to predict the motion
% vector. First, Initialize the parameters for computing the motion vector.
[h, w]=size(im0);
[bh, bw]=size(blk); 
mv0.x=0;
mv0.y=0;
mv1.x=0;
mv1.y=0;
mad0 = inf*ones(2*range+1, 2*range+1);
mad_min0 =inf; 
mad1 = inf*ones(2*range+1, 2*range+1);
mad_min1 =inf; 
j = 1; 
sadCount = 0;
% Loop over the x range and the y range.
for y_offs=-range:range
    k = 1;
    for x_offs = -range:range
        % Determine if the block-under-test is in the image and the block.
        inBlk = y0+y_offs > 0 && y0+y_offs <= (h-bh) && ...
            x0+x_offs > 0 && x0+x_offs <= (w-bw);

        if inBlk
            % Calculate a prediction using the previous frame.
            blk_res0 = abs(im0(y0+y_offs:y0+y_offs+bh-1, ...
                x0+x_offs: x0+x_offs+bw-1) - blk);

            % Compute the mean absolute deviation of the residual block as
            % well as the motion vector.
            mad0(j,k) = mean(blk_res0(:));
            sadCount = sadCount + 1;
            mvs0(j,k).x = x_offs;
            mvs0(j,k).y = y_offs;

            % If the mean absolute deviation is the minimum, capture the
            % information of the block.
            if mad0(j,k) < mad_min0
                mad_min0 = mad0(j,k); 
                mv0 = mvs0(j,k); 
                res0 = blk_res0; 
            end
            
            % Calculate a prediction using the next frame.
            blk_res1 = abs(im1(y0+y_offs:y0+y_offs+bh-1, ...
                x0+x_offs: x0+x_offs+bw-1) - blk);

            % Compute the mean absolute deviation of the residual block as
            % well as the motion vector.
            mad1(j,k) = mean(blk_res1(:)); 
            sadCount = sadCount + 1;
            mvs1(j,k).x = x_offs;
            mvs1(j,k).y = y_offs;

            % If the mean absolute deviation is the minimum, capture the
            % information of the block.
            if mad1(j,k) < mad_min1
                mad_min1 = mad1(j,k); 
                mv1 = mvs1(j,k); 
                res1 = blk_res1; 
            end
        else
            % If the mean absolute deviation of the residual block is not
            % in the block-under-test, move on.
            mad0(j,k) = inf;
            mvs0(j,k).x = x_offs;
            mvs0(j,k).y = y_offs;
            
            mad1(j,k) = inf; 
            mvs1(j,k).x = x_offs;
            mvs1(j,k).y = y_offs;
        end
        k = k + 1;
    end
    j = j + 1;
end

% The residual block is the average of the previous and next frame.
res = (res0/2 + res1/2);
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
