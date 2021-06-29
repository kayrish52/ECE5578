% This is the top level script for ECE5578 Homework 1, Problem 3.

% Clear the workspace.
clear all;

% Initialize the image file.
im = 'Lena.png';

% Read the image.
im = imread(im);

% Convert to Grayscale.
im = rgb2gray(im);

% Initialize the filter.
W = [0.333, 0.333, 0; 0.333, 0, 0; 0, 0, 0];

% Filter the image to a residual image.
imFilt = imfilter(im,W,'replicate');

% Compute the residual.
imRes = double(imFilt)-double(im);

% Initialize the centers of bins for histogram.
bounds = -255:255;
plotBounds = -30:30;

% Create the histogram.
imHistCounts = histcounts(imRes,bounds);
pImCounts = imHistCounts/sum(imHistCounts(:));

% Generate a doublesided Geometric Distribution.
prob = genGeomDist(0.81);

% Plot the histogram.
hold off;
figure(1);
imHist = histogram(imRes,plotBounds);
xticks([-30:5:30]);
xticklabels({'-30','-25','-20','-15','-10','-5','0','5','10',...
    '15','20','25','30'});
xlabel('Histogram Bin Centers')
ylabel('Counts')
title('Residual Distribution of lena.png')

% Plot the distribution of residuals.
figure(2);
hold on;
plot(-30:30,pImCounts(226:286),'g',-30:30,prob,'r')
xticks([-30:5:30]);
xticklabels({'-30','-25','-20','-15','-10','-5','0','5','10',...
    '15','20','25','30'});
xlabel('Residual Value')
ylabel('Probability Value')
title('Probability Distribution of the Residual of Lena.png')
legend('Residuals','Geometric Distribution of rho = 0.81',...
    'Location','SouthOutside')

function [prob] = genGeomDist(rho)
% genGeomDist generates a doublesided geometric distribution for the input 
% value of rho.

% Initialize the line.
x = -30:30;

% Generate the probability.
prob = (1-rho)/(1+rho)*power(rho,abs(x));

% Plot the probability.
figure(3)
plot(x,prob)
xticks([-30:5:30]);
xticklabels({'-30','-25','-20','-15','-10','-5','0','5','10',...
    '15','20','25','30'});
xlabel('Residual Value')
ylabel('Probability')
title('Doublesided Geometric Distribution Estimation of Lena.png')
end


