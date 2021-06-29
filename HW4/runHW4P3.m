% This script will retrieve the original frame data 'Beauty.mp4' and 
% 'Bosphorus.mp4'. Then it will display the SSIM map for analysis.
% analyzed.

% Close all images.
clear all;
close all;

% Read the images of 'Beauty' into memory.
beautyO = imread('Data/Beauty/BeautyOriginal2.jpg');
beautyI = imread('Data/Beauty/BeautyI2.jpg');
beautyP = imread('Data/Beauty/BeautyP2.jpg');

% Compute the Structural Similarity Index Measure.
[ssimvalI,ssimmapI] = ssim(beautyI,beautyO);
[ssimvalP,ssimmapP] = ssim(beautyP,beautyO);
[ssimvalC,ssimmapC] = ssim(beautyP,beautyI);

% Display the SSIM Maps for the three combinations.
figure(1)
subplot (1,3,1)
imshow(ssimmapI)
title({sprintf('SSIM = %1.3f',ssimvalI),'Frame 2: INTRA vs Original'})
subplot (1,3,2)
imshow(ssimmapP)
title({sprintf('SSIM = %1.3f',ssimvalP),'Frame 2: Predicted vs Original'})
subplot (1,3,3)
imshow(ssimmapC)
title({sprintf('SSIM = %1.3f',ssimvalC),'Frame 2: INTRA vs Predicted'})

% Read the images of 'Bosphorus' into memory.
bosO = imread('Data/Bosphorus/BosphorusOriginal2.jpg');
bosI = imread('Data/Bosphorus/BosphorusI2.jpg');
bosP = imread('Data/Bosphorus/BosphorusP2.jpg');

% Compute the Structural Similarity Index Measure.
[ssimvalI,ssimmapI] = ssim(bosI,bosO);
[ssimvalP,ssimmapP] = ssim(bosP,bosO);
[ssimvalC,ssimmapC] = ssim(bosP,bosI);

% Display the SSIM Maps for the three combinations.
figure(2)
subplot (1,3,1)
imshow(ssimmapI)
title({sprintf('SSIM = %1.3f',ssimvalI),'Frame 2: INTRA vs Original'})
subplot (1,3,2)
imshow(ssimmapP)
title({sprintf('SSIM = %1.3f',ssimvalP),'Frame 2: Predicted vs Original'})
subplot (1,3,3)
imshow(ssimmapC)
title({sprintf('SSIM = %1.3f',ssimvalC),'Frame 2: INTRA vs Predicted'})
