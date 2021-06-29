% This is the main script for ECE5578, Final Project, Problem 3.
% Clear the workspace.
clear all;
close all;

% Load VGG Conv2 features: 128 channels of 128x128, for 100 images from VOC
% data set: features
load('Features/vgg16_block2_conv2_features.mat');

% Capture the features of the images.
boatFeat = features(1,:);
bottleFeat = features(11,:);
catFeat = features(21,:);
chairFeat = features(31,:);

% Initialize, split, and normalize the channels of each feature set.
boatFeats = cell(15,1);
bottleFeats = cell(15,1);
catFeats = cell(15,1);
chairFeats = cell(15,1);
for i = 1:15
    range = i:128:(power(128,3)-128+i);

    % Process the feature maps.
    boatFeats{i,1} = reshape(boatFeat(1,range),[128,128]);
    bottleFeats{i,1} = reshape(bottleFeat(1,range),[128,128]);
    catFeats{i,1} = reshape(catFeat(1,range),[128,128]);
    chairFeats{i,1} = reshape(chairFeat(1,range),[128,128]);
end

% Initialize and loop over each frame to determine the best fit reference
% frame for Motion Estimation.
boatBestFitFrame = inf*ones(15,1);
boatBestFitMSE = inf*ones(15,1);
for i = 1:15
    plt.plt = 0;
    plt.path = 'DataImages/Problem3/Boat/boat';
    plt.frame = i;
    plt.fit = 0;
    plt.mse = inf;
    for j = 1:15
        if i == j, continue; end
        [mse,~] = estMotion(boatFeats{i,1},boatFeats{j,1},plt);
        if mse < boatBestFitMSE(i,1)
            boatBestFitMSE(i,1) = mse;
            boatBestFitFrame(i,1) = j;
            plt.fit = j;
            plt.mse = mse;
        end
        fprintf('Do not worry, you are still sane! i = %i, j = %i\n',i,j);
    end
    
    % For the best fit, set the plotting feature and recall 'estMotion'.
    plt.plt = 1;
    [~,~] = estMotion(boatFeats{i,1},boatFeats{plt.fit,1},plt);
end

% Initialize and loop over each frame to determine the best fit reference
% frame for Motion Estimation.
bottleBestFitFrame = inf*ones(15,1);
bottleBestFitMSE = inf*ones(15,1);
for i = 1:15
    plt.plt = 0;
    plt.path = 'DataImages/Problem3/Bottle/bottle';
    plt.frame = i;
    plt.fit = 0;
    plt.mse = inf;
    for j = 1:15
        if i == j, continue; end
        [mse,~] = estMotion(bottleFeats{i,1},bottleFeats{j,1},plt);
        if mse < bottleBestFitMSE(i,1)
            bottleBestFitMSE(i,1) = mse;
            bottleBestFitFrame(i,1) = j;
            plt.fit = j;
            plt.mse = mse;
        end
        fprintf('Do not worry, you are still sane! i = %i, j = %i\n',i,j);
    end
    
    % For the best fit, set the plotting feature and recall 'estMotion'.
    plt.plt = 1;
    [~,~] = estMotion(bottleFeats{i,1},bottleFeats{plt.fit,1},plt);
end

% Initialize and loop over each frame to determine the best fit reference
% frame for Motion Estimation.
catBestFitFrame = inf*ones(15,1);
catBestFitMSE = inf*ones(15,1);
for i = 1:15
    plt.plt = 0;
    plt.path = 'DataImages/Problem3/Cat/cat';
    plt.frame = i;
    plt.fit = 0;
    plt.mse = inf;
    for j = 1:15
        if i == j, continue; end
        [mse,~] = estMotion(catFeats{i,1},catFeats{j,1},plt);
        if mse < catBestFitMSE(i,1)
            catBestFitMSE(i,1) = mse;
            catBestFitFrame(i,1) = j;
            plt.fit = j;
            plt.mse = mse;
        end
        fprintf('Do not worry, you are still sane! i = %i, j = %i\n',i,j);
    end
    
    % For the best fit, set the plotting feature and recall 'estMotion'.
    plt.plt = 1;
    [~,~] = estMotion(catFeats{i,1},catFeats{plt.fit,1},plt);
end

% Initialize and loop over each frame to determine the best fit reference
% frame for Motion Estimation.
chairBestFitFrame = inf*ones(15,1);
chairBestFitMSE = inf*ones(15,1);
for i = 1:15
    plt.plt = 0;
    plt.path = 'DataImages/Problem3/Chair/chair';
    plt.frame = i;
    plt.fit = 0;
    plt.mse = inf;
    for j = 1:15
        if i == j, continue; end
        [mse,~] = estMotion(chairFeats{i,1},chairFeats{j,1},plt);
        if mse < chairBestFitMSE(i,1)
            chairBestFitMSE(i,1) = mse;
            chairBestFitFrame(i,1) = j;
            plt.fit = j;
            plt.mse = mse;
        end
        fprintf('Do not worry, you are still sane! i = %i, j = %i\n',i,j);
    end
    
    % For the best fit, set the plotting feature and recall 'estMotion'.
    plt.plt = 1;
    [~,~] = estMotion(chairFeats{i,1},chairFeats{plt.fit,1},plt);
end

% Save the data in a '.mat' file.
save('DataImages/Problem3/mseData.mat',...
    'boatBestFitMSE','boatBestFitFrame',...
    'bottleBestFitMSE','bottleBestFitFrame',...
    'catBestFitMSE','catBestFitFrame',...
    'chairBestFitMSE','chairBestFitFrame')
return;

% This function will estimate the motion vector...
function [mse,sadCounter] = estMotion(im1,imPrev,plt)

% This will loop over the image, splitting it into blocks of size 8x8, and
% calling the function 'getBlkMotion' to estimate the block motion.
blkSize = 4;
[h,w,~] = size(im1);
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
        imBlocks{y,x} = im1(yRange,xRange);
    end
end

% This loop will comput ethe motion vectors and the residual block. First,
% initialize the motion estimation range, as well as the pel value.
pel = 0.5;
if or(pel == 0.5,pel == 0.25)
    im1 = bilinearInterpolation(im1, (1/pel)*[h, w]);
    imPrev = bilinearInterpolation(imPrev, (1/pel)*[h, w]);
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
            getBlkMotion(imBlocks{y,x},xLoc,yLoc,imPrev,pel,range);
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
        mvPlot.xLoc(y,x) = xLoc;
        mvPlot.yLoc(y,x) = yLoc;
        mvPlot.x(y,x) = mvBlks{y,x}.x;
        mvPlot.y(y,x) = mvBlks{y,x}.y;
    end
end

% If the plot feature is set, plot the estimated frame and the quiver plot.
if plt.plt
    fig1 = figure('visible','off');
    sgtitle({['Input Frame: ',num2str(plt.frame)],...
        ['Best Motion Estimation Frame: ',num2str(plt.fit)],...
        ['Mean Square Error: ',num2str(plt.mse)]});
    subplot(2,3,1)
    imagesc(im1)
    colormap('gray')
    title('Predicted Feature Map')
    
    subplot(2,3,4)
    imagesc(imPrev)
    colormap('gray')
    title('Input Feature Map')
    
    subplot(2,3,2)
    imagesc(double(abs(imPrev - im1)))
    colormap('gray')
    title({'Actual','Residual'})
    
    subplot(2,3,5)
    imagesc(resImg)
    colormap('gray')
    title({'Predicted','Residual'})
    
    subplot(2,3,3)
    imagesc(im1 + double(abs(imPrev - im1)))
    colormap('gray')
    title({'Actual','Reconstructed'})
    
    subplot(2,3,6)
    imagesc(im1 + resImg)
    colormap('gray')
    title({'Predicted','Reconstructed'})
    
    % Set the save path and save the figure.
    savePath = strcat(plt.path,'Frame',num2str(plt.frame),'_BestFit',...
        num2str(plt.fit),'_');
    saveas(fig1,strcat(savePath,'MotionEstimation.jpg'));

    % Plot the quiver plot and save the figure.
    fig2 = figure('visible','off');
    imagesc(im1); hold on; grid on; colormap('gray');
    quiver(mvPlot.xLoc(:),mvPlot.yLoc(:),mvPlot.x(:),mvPlot.y(:), '-r');
    title({'Motion Vectors',['Input Frame: ',num2str(plt.frame)],...
        ['Best Motion Estimation Frame: ',num2str(plt.fit)],...
        ['Mean Square Error: ',num2str(plt.mse)]});
    saveas(fig2,strcat(savePath,'Quiver.jpg'));
end

% Compute the Mean Square Error of the Residual.
mse = 0;
for i = 1:h
    for j = 1:w
        mse = mse + power(resImg(i,j),2);
    end
end
mse = mse/(w*h);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% THE FOLLOWING FUNCTIONS SUPPORT PROBLEM 3 OF THE FINAL PROJECT %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the function provided by Dr. Li to estimate the block motion and
% the residual.
function [mv,res,sadCount]=getBlkMotion(blk, x0 , y0, im, pel, range)
% INPUTS:
%  - blk - m x m blocks
%  - x0, y0 - blk location in current frame
%  - im - hxw a frame
%  - pel = 1, 0.5 0.25: pel steps
%  - range = 8; 
% output:
%   mv - motion vector
%   res - residual.

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
    
% Following Bilinear Interpolation, update the image and block size.
[h, w]=size(im);
[bh, bw]=size(blk);

% Initialize the motion vector.
mv.x=0;
mv.y=0;

% This loop will sweep over the range for each block to predict the motion
% vector. First, Initialize the parameters for computing the motion vector.
mad = inf*ones(2*range+1, 2*range+1);
mad_min=inf;
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
            % Capture the residual of the block.
            blk_res = double(abs(im(y0+y_offs:y0+y_offs+bh-1, ...
                x0+x_offs: x0+x_offs+bw-1) - blk));
            
            % Compute the mean absolute deviation of the residual block.
            mad(j,k) = mean(blk_res(:));
            sadCount = sadCount + 1;
            
            % If the mean absolute deviation of the residual block is the
            % minimum, save the motion vector as the 'x' and 'y' offsets,
            % and the residual block.
            mvs(j,k).x = x_offs;
            mvs(j,k).y = y_offs;
            if mad(j,k) < mad_min
                mad_min = mad(j,k);
                mv = mvs(j,k); 
                res = blk_res;
            end
        else
            % If the mean absolute deviation of the residual block is not
            % in the block-under-test, move on.
            mad(j,k) = inf;
            sadCount = sadCount + 1;
            mvs(j,k).x = x_offs; mvs(j,k).y = y_offs;
        end
        k = k + 1;
    end
    j = j + 1;
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

