% This is the main script for Problem 2 of the Final Project for ECE5578 
% for the Spring 2020 Semester.

% Clear the workspace. Close all figures.
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
boatFeats = cell(8,1);
bottleFeats = cell(8,1);
catFeats = cell(8,1);
chairFeats = cell(8,1);
for i = 1:8
    % Compute the range.
    range = i:128:(power(128,3)-128+i);

    % Process the feature maps.
    boatFeats{i,1} = rescale(reshape(boatFeat(1,range),[128,128]),0,255);
    bottleFeats{i,1} = rescale(reshape(bottleFeat(1,range),[128,128]),0,255);
    catFeats{i,1} = rescale(reshape(catFeat(1,range),[128,128]),0,255);
    chairFeats{i,1} = rescale(reshape(chairFeat(1,range),[128,128]),0,255);
end

% Initialize the filter.
W = [0.333, 0.333, 0; 0.333, 0, 0; 0, 0, 0];

% Filter each image feature to a residual image.
boatFilt = cell(8,1);
bottleFilt = cell(8,1);
catFilt = cell(8,1);
chairFilt = cell(8,1);
for i = 1:8
    boatFilt{i,1} = double(imfilter(boatFeats{i,1},W,'replicate'));
    bottleFilt{i,1} = double(imfilter(bottleFeats{i,1},W,'replicate'));
    catFilt{i,1} = double(imfilter(catFeats{i,1},W,'replicate'));
    chairFilt{i,1} = double(imfilter(chairFeats{i,1},W,'replicate'));
end

% Create the histograms.
boatHist = cell(8,6);
boatBins = cell(8,6);
boatEnt = zeros(8,6);
bottleHist = cell(8,6);
bottleBins = cell(8,6);
bottleEnt = zeros(8,6);
catHist = cell(8,6);
catBins = cell(8,6);
catEnt = zeros(8,6);
chairHist = cell(8,6);
chairBins = cell(8,6);
chairEnt = zeros(8,6);
q = [0,1,2,4,8,16];
for i = 1:8
    for j = 1:length(q)
        
        % If j = 1, do not quantize the data. Else, use quantization.
        if j == 1
            boatHist{i,j} = ...
                histcounts(boatFeats{i,1},linspace(0,255,256))/(128*128);
            boatHist{i,j} = boatHist{i,j}/sum(boatHist{i,j}(:));
            boatBins{i,j} = linspace(0,255,256)';
            boatEnt(i,j) = getEntropy(boatHist{i,j});
            bottleHist{i,j} = ...
                histcounts(bottleFeats{i,1},linspace(0,255,256))/(128*128);
            bottleHist{i,j} = bottleHist{i,j}/sum(bottleHist{i,j}(:));
            bottleBins{i,j} = linspace(0,255,256)';
            bottleEnt(i,j) = getEntropy(bottleHist{i,j});
            catHist{i,j} = ...
                histcounts(catFeats{i,1},linspace(0,255,256))/(128*128);
            catHist{i,j} = catHist{i,j}/sum(catHist{i,j}(:));
            catBins{i,j} = linspace(0,255,256)';
            catEnt(i,j) = getEntropy(catHist{i,j});
            chairHist{i,j} = ...
                histcounts(chairFeats{i,1},linspace(0,255,256))/(128*128);
            chairHist{i,j} = chairHist{i,j}/sum(chairHist{i,j}(:));
            chairBins{i,j} = linspace(0,255,256)';
            chairEnt(i,j) = getEntropy(chairHist{i,j});
        else
            [~,~,~,~,boatHist{i,j},boatBins{i,j},boatEnt(i,j)] = ...
                imQuantCalc(boatFeats{i,1},boatFilt{i,1},q(j));
            [~,~,~,~,bottleHist{i,j},bottleBins{i,j},bottleEnt(i,j)] = ...
                imQuantCalc(bottleFeats{i,1},bottleFilt{i,1},q(j));
            [~,~,~,~,catHist{i,j},catBins{i,j},catEnt(i,j)] = ...
                imQuantCalc(catFeats{i,1},catFilt{i,1},q(j));
            [~,~,~,~,chairHist{i,j},chairBins{i,j},chairEnt(i,j)] = ...
                imQuantCalc(chairFeats{i,1},chairFilt{i,1},q(j));
        end
        fprintf('Do not worry. You are still sane. i = %i, j = %i\n',i,j);
    end
end

% Plot the histograms.
plotHist(boatBins,boatHist,boatEnt,'Boat');
plotHist(bottleBins,bottleHist,bottleEnt,'Bottle');
plotHist(catBins,catHist,catEnt,'Cat');
plotHist(chairBins,chairHist,chairEnt,'Chair');

return;

% This starts the imQuant function.
function [imQ,h1,cBins1,eO,h2,cBins2,eQ] = imQuantCalc(im,imFilt,q)

% Quantize the residual image.
imQ = round(imFilt/q)*q;

% Determine the bin boundaries.
pos = 0;
cBounds = zeros(1,1);
cBounds(1,1) = pos;
while pos <= 255
    pos = pos + q;
    cBounds(end+1,1) = pos;
end

% Compute histogram data.
[hist1,cBins1] = histcounts(im, linspace(0,255,256));
[hist2,cBins2] = histcounts(imQ(:), cBounds);

% Normalize the histogram data.
h1 = hist1./sum(hist1(:));
h2 = hist2./sum(hist2(:));

% Compute the Engropy of each image.
eO = getEntropy(h1);
eQ = getEntropy(h2);

end

% This function will receive a set of histograms, bins, and entropys and
% generate a series of plots for each feature map.
function [] = plotHist(hBins,hPMF,ent,figName)

% Set the step sizes.
step = [0.5,0.5,1,2,4,8];

% For each feature map, loop over and plot each histogram as the
% quantization level varies.
for i = 1:8
    % Set the figure title.
    fig = figure('visible','off');
    sgtitle(['Histograms for ',figName,', Feature Map #',num2str(i)])

    % Update the bounds, and plot the 'No Quantization' case.
    plotBounds = hBins{i,1} + step(1);
    plotBounds(end,:) = [];
    subplot(2,3,1)
    bar(plotBounds,hPMF{i,1})
    title({['Histogram of ',figName],...
        sprintf('Feature Map %i',i),...
        'No Quantization',...
        sprintf('Entropy = %1.2f',ent(i,1))})

    % Update the bounds, and plot the 'Q = 1' Quantization case.
    plotBounds = hBins{i,2} + step(2);
    plotBounds(end,:) = [];
    subplot(2,3,2)
    bar(plotBounds,hPMF{i,2})
    title({['Histogram of ',figName],...
        sprintf('Feature Map %i',i),...
        'Quantization: q = 1',...
        sprintf('Entropy = %1.2f',ent(i,2))})

    % Update the bounds, and plot the 'Q = 2' Quantization case.
    plotBounds = hBins{i,3} + step(3);
    plotBounds(end,:) = [];
    subplot(2,3,3)
    bar(plotBounds,hPMF{i,3})
    title({['Histogram of ',figName],...
        sprintf('Feature Map %i',i),...
        'Quantization: q = 2',...
        sprintf('Entropy = %1.2f',ent(i,3))})

    % Update the bounds, and plot the 'Q = 4' Quantization case.
    plotBounds = hBins{i,4} + step(4);
    plotBounds(end,:) = [];
    subplot(2,3,4)
    bar(plotBounds,hPMF{i,4})
    title({['Histogram of ',figName],...
        sprintf('Feature Map %i',i),...
        'Quantization: q = 4',...
        sprintf('Entropy = %1.2f',ent(i,4))})
    
    % Update the bounds, and plot the 'Q = 8' Quantization case.
    plotBounds = hBins{i,5} + step(5);
    plotBounds(end,:) = [];
    subplot(2,3,5)
    bar(plotBounds,hPMF{i,5})
    title({['Histogram of ',figName],...
        sprintf('Feature Map %i',i),...
        'Quantization: q = 8',...
        sprintf('Entropy = %1.2f',ent(i,5))})
    
    % Update the bounds, and plot the 'Q = 16' Quantization case.
    plotBounds = hBins{i,6} + step(6);
    plotBounds(end,:) = [];
    subplot(2,3,6)
    bar(plotBounds,hPMF{i,6})
    title({['Histogram of ',figName],...
        sprintf('Feature Map %i',i),...
        'Quantization: q = 16',...
        sprintf('Entropy = %1.2f',ent(i,6))})

    % Save the histogram figure.
    saveas(fig,['DataImages/Problem2/',figName,'/',...
        figName,sprintf('_FeatureMap%i',i),'Histogram.jpg'])
end
end

% This starts the getEntropy function.
function [e] = getEntropy(pmf)

% The Probability Mass Function requires positive data. This isolates
% values that are greater than zero.
pmf = pmf(pmf>0);

% Compute the Entropy of the Probability Mass Function.
e = -1*sum(pmf.*log(pmf));
end

