% This is the main script for Problem 1 of the Final Project of ECE5578 
% for the Spring 2020 Semester. This script will execute all of the 
% problems for the final project.

% Clear the workspace. Close all figures.
clear all;
close all;

% Read the images into a cell array.
ims = cell(4,1);
ims{1,1} = imread('Images/boat/2008_001047.jpg');
ims{2,1} = imread('Images/bottle/2008_001744.jpg');
ims{3,1} = imread('Images/cat/2008_002793.jpg');
ims{4,1} = imread('Images/chair/2008_002148.jpg');

% Plot the original Images.
% Boat Image.
f = figure('visible','off');
subplot(2,2,1);
imagesc(ims{1,1});
title('Image #1, Boat')

% Bottle Image.
subplot(2,2,2);
imagesc(ims{2,1});
title('Image #2, Bottle')

% Cat Image.
subplot(2,2,3);
imagesc(ims{3,1});
title('Image #3, Cat')

% Chair Image.
subplot(2,2,4);
imagesc(ims{4,1});
title('Image #4, Chair')

% Save the image.
saveas(f,'DataImages/OriginalImages.jpg');
fprintf('Original Images Saved!\n');

% Create a Message Box alerting the user of the need to press a key before
% moving forward.
msg = msgbox('Press Any Key to Continue','ALERT!');
msgPos = get(msg,'Position');
set(msg,'Position', msgPos+[0,0,10,0])
uiwait(msg,1);
delete(msg);
close all;
clearvars msg f msgPos;

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
    range = i:128:(power(128,3)-128+i);

    % Process the feature maps.
    boatFeats{i,1} = reshape(boatFeat(1,range),[128,128]);
    bottleFeats{i,1} = reshape(bottleFeat(1,range),[128,128]);
    catFeats{i,1} = reshape(catFeat(1,range),[128,128]);
    chairFeats{i,1} = reshape(chairFeat(1,range),[128,128]);
end
fprintf('Feature Maps Extracted!\n');

% Visualize the first 8 feature maps for each image.
for i=1:8
    % Set the figure title.
    if i == 1
        f1 = figure('visible','off');
        sgtitle('First 8 Feature Maps for Boat');
        f2 = figure('visible','off');
        sgtitle('First 8 Feature Maps for Bottle');
        f3 = figure('visible','off');
        sgtitle('First 8 Feature Maps for Cat');
        f4 = figure('visible','off');
        sgtitle('First 8 Feature Maps for Chair');
    end
    
    % First 8 feature maps of the Boat.
    figure(f1);
    set(f1,'visible','off');
    x = boatFeats{i,1};
    subplot(2,4,i);
    imagesc(x);
    title({'CONV2D',sprintf('Feature Map #%d',i)});
    axis off;
    
    % First 8 feature maps of the Bottle.
    figure(f2);
    set(f2,'visible','off');
    x = bottleFeats{i,1};
    subplot(2,4,i);
    imagesc(x);
    title({'CONV2D',sprintf('Feature Map #%d',i)});
    axis off;

    % First 8 feature maps of the Cat.
    figure(f3);
    set(f3,'visible','off');
    x = catFeats{i,1};
    subplot(2,4,i);
    imagesc(x);
    title({'CONV2D',sprintf('Feature Map #%d',i)});
    axis off;

    % First 8 feature maps of the Chair.
    figure(f4);
    set(f4,'visible','off');
    x = chairFeats{i,1};
    subplot(2,4,i);
    imagesc(x);
    title({'CONV2D',sprintf('Feature Map #%d',i)});
    axis off;
end
% Save the image.
saveas(f1,'DataImages/Problem1/Boat/BoatFeatureMaps.jpg');
saveas(f2,'DataImages/Problem1/Bottle/BottleFeatureMaps.jpg');
saveas(f3,'DataImages/Problem1/Cat/CatFeatureMaps.jpg');
saveas(f4,'DataImages/Problem1/Chair/ChairFeatureMaps.jpg');
clearvars f1 f2 f3 f4;
fprintf('Feature Maps Printed!\n')

% Initialize the Discrete Cosine Transform cell arrays.
boatDCT = cell(8,1);
bottleDCT = cell(8,1);
catDCT = cell(8,1);
chairDCT = cell(8,1);

% Compute the Discrete Cosine Transform (DCT) of each feature map.
for i = 1:8
    % Compute the DCTs of the images.
    boatDCT{i,1} = dct2(boatFeats{i,1});
    bottleDCT{i,1} = dct2(bottleFeats{i,1});
    catDCT{i,1} = dct2(catFeats{i,1});
    chairDCT{i,1} = dct2(chairFeats{i,1});
end

% Visualize the first 8 feature map DCTs for each image.
for i=1:8
    % Set the figure title.
    if i == 1
        f1 = figure;
        sgtitle('First 8 Feature Map DCTs for Boat');
        f2 = figure;
        sgtitle('First 8 Feature Map DCTs for Bottle');
        f3 = figure;
        sgtitle('First 8 Feature Map DCTs for Cat');
        f4 = figure;
        sgtitle('First 8 Feature Map DCTs for Chair');
    end
    
    % First 8 feature map DCTs of the Boat.
    figure(f1);
    x = boatDCT{i,1};
    subplot(2,4,i);
    imagesc(x);
    title({'DCT',sprintf('Feature Map #%d',i)});
    axis off;
    
    % First 8 feature map DCTs of the Bottle.
    figure(f2);
    x = bottleDCT{i,1};
    subplot(2,4,i);
    imagesc(x);
    title({'DCT',sprintf('Feature Map #%d',i)});
    axis off;

    % First 8 feature map DCTs of the Cat.
    figure(f3);
    x = catDCT{i,1};
    subplot(2,4,i);
    imagesc(x);
    title({'DCT',sprintf('Feature Map #%d',i)});
    axis off;

    % First 8 feature map DCTs of the Chair.
    figure(f4);
    x = chairDCT{i,1};
    subplot(2,4,i);
    imagesc(x);
    title({'DCT',sprintf('Feature Map #%d',i)});
    axis off;
end
% Save the image.
saveas(f1,'DataImages/Problem1/Boat/BoatDCT.jpg');
saveas(f2,'DataImages/Problem1/Bottle/BottleDCT.jpg');
saveas(f3,'DataImages/Problem1/Cat/CatDCT.jpg');
saveas(f4,'DataImages/Problem1/Chair/ChairDCT.jpg');
clearvars f1 f2 f3 f4;
fprintf('DCT Figures Printed!\n')

% For each DCT Feature Map, save a series of JPEG images of varying
% quality.
for i = 1:8
    % Vary the range of Quality of Images.    
    for j = 0:10:100
        % Save the Boat DCT Feature Maps.
        fileName = sprintf(...
            'DataImages/Problem1/Boat/FeatMap%d_Qual%d.jpeg',i,j);
        imwrite(boatDCT{i,1},fileName,'JPEG','quality',j,'bitdepth',8)

        % Save the Bottle DCT Feature Maps.
        fileName = sprintf(...
            'DataImages/Problem1/Bottle/FeatMap%d_Qual%d.jpeg',i,j);
        imwrite(bottleDCT{i,1},fileName,'JPEG','quality',j,'bitdepth',8)

        % Save the Cat DCT Feature Maps.
        fileName = sprintf(...
            'DataImages/Problem1/Cat/FeatMap%d_Qual%d.jpeg',i,j);
        imwrite(catDCT{i,1},fileName,'JPEG','quality',j,'bitdepth',8)

        % Save the Chair DCT Feature Maps.
        fileName = sprintf(...
            'DataImages/Problem1/Chair/FeatMap%d_Qual%d.jpeg',i,j);
        imwrite(chairDCT{i,1},fileName,'JPEG','quality',j,'bitdepth',8)
    end
    fprintf(['Quality Images for Feature Map ',num2str(i),' Created!\n']);
end

% For each level of Image Quality, Reconstruct the Image and Compute the
% PSNR. Initialize the Reconstruction variables.
boatRecon = cell(8,11);
bottleRecon = cell(8,11);
catRecon = cell(8,11);
chairRecon = cell(8,11);
boatReconDCT = cell(8,11);
bottleReconDCT = cell(8,11);
catReconDCT = cell(8,11);
chairReconDCT = cell(8,11);
boatPSNR = zeros(8,11);
bottlePSNR = zeros(8,11);
catPSNR = zeros(8,11);
chairPSNR = zeros(8,11);
boatBPP = zeros(8,11);
bottleBPP = zeros(8,11);
catBPP = zeros(8,11);
chairBPP = zeros(8,11);
for i = 1:8
    % Set the image counter.
    k = 1;
    
    % Vary the range of Quality of Images.
    for j = 0:10:100
        % Load the DCT Feature Maps and Capture the bits per pixel.
        fileName = sprintf(...
            'DataImages/Problem1/Boat/FeatMap%d_Qual%d.jpeg',i,j);
        imInf = imfinfo(fileName,'JPEG');
        boatBPP(i,k) = double((imInf.FileSize*8)/(128*128));
        boatReconDCT{i,k} = imread(fileName);

        fileName = sprintf(...
            'DataImages/Problem1/Bottle/FeatMap%d_Qual%d.jpeg',i,j);
        imInf = imfinfo(fileName,'JPEG');
        bottleBPP(i,k) = double((imInf.FileSize*8)/(128*128));
        bottleReconDCT{i,k} = imread(fileName);
        
        fileName = sprintf(...
            'DataImages/Problem1/Cat/FeatMap%d_Qual%d.jpeg',i,j);
        imInf = imfinfo(fileName,'JPEG');
        catBPP(i,k) = double((imInf.FileSize*8)/(128*128));
        catReconDCT{i,k} = imread(fileName);

        fileName = sprintf(...
            'DataImages/Problem1/Chair/FeatMap%d_Qual%d.jpeg',i,j);
        imInf = imfinfo(fileName,'JPEG');
        chairBPP(i,k) = double((imInf.FileSize*8)/(128*128));
        chairReconDCT{i,k} = imread(fileName);
        
        % Compute the Inverse DCT.
        boatRecon{i,k} = idct2(boatReconDCT{i,k});
        bottleRecon{i,k} = idct2(bottleReconDCT{i,k});
        catRecon{i,k} = idct2(catReconDCT{i,k});
        chairRecon{i,k} = idct2(chairReconDCT{i,k});
        
        % Compute the PSNR Relative to the original.
        boatRecon{i,k}(1:4,:) = [];
        boatRecon{i,k}(:,1:4) = [];
        boatPSNR(i,k) = psnr(boatRecon{i,k},boatFeats{i,1}(5:end,5:end));
        bottleRecon{i,k}(1:4,:) = [];
        bottleRecon{i,k}(:,1:4) = [];
        bottlePSNR(i,k) = psnr(bottleRecon{i,k},bottleFeats{i,1}(5:end,5:end));
        catRecon{i,k}(1:4,:) = [];
        catRecon{i,k}(:,1:4) = [];
        catPSNR(i,k) = psnr(catRecon{i,k},catFeats{i,1}(5:end,5:end));
        chairRecon{i,k}(1:4,:) = [];
        chairRecon{i,k}(:,1:4) = [];
        chairPSNR(i,k) = psnr(chairRecon{i,k},chairFeats{i,1}(5:end,5:end));

        % Increment the image counter.
        k = k + 1;
    end
    
    fprintf(['PSNR for Feature Map ',num2str(i),' Computed!\n']);
end

% Plot the PSNR vs BPP R-D Curve for the first 8 feature maps.
j = 0:10:100;
for i = 1:8
    % Set the figure title.
    if i == 1
        f1 = figure('visible','off');
        sgtitle({'PSNR vs BPP for Boat','First 8 Feature Maps'})
        f2 = figure('visible','off');
        sgtitle({'PSNR vs BPP for Bottle','First 8 Feature Maps'})
        f3 = figure('visible','off');
        sgtitle({'PSNR vs BPP for Cat','First 8 Feature Maps'})
        f4 = figure('visible','off');
        sgtitle({'PSNR vs BPP for Chair','First 8 Feature Maps'})
    end
    
    
    % Plot the PSNR vs BPP for Boat.
    figure(f1);
    set(f1,'visible','off')
    subplot(2,4,i);
    stairs(boatBPP(i,:),abs(boatPSNR(i,:)),'LineWidth',1)
    title(sprintf('Feature Map #%i',i))
    xlabel('Bits Per Pixel');
    ylabel('PSNR (dB)')
    hold on;
    scatter(boatBPP(i,:),abs(boatPSNR(i,:)),'o')

    % Plot the PSNR vs BPP for Bottle.
    figure(f2);
    set(f2,'visible','off')
    sgtitle({'PSNR vs BPP for Bottle','First 8 Feature Maps'})
    subplot(2,4,i);
    stairs(bottleBPP(i,:),abs(bottlePSNR(i,:)),'LineWidth',1)
    title(sprintf('Feature Map #%i',i))
    xlabel('Bits Per Pixel');
    ylabel('PSNR (dB)')
    hold on;
    scatter(bottleBPP(i,:),abs(bottlePSNR(i,:)),'o')

    % Plot the PSNR vs BPP for Cat.
    figure(f3);
    set(f3,'visible','off')
    subplot(2,4,i);
    stairs(catBPP(i,:),abs(catPSNR(i,:)),'LineWidth',1)
    title(sprintf('Feature Map #%i',i))
    xlabel('Bits Per Pixel');
    ylabel('PSNR (dB)')
    hold on;
    scatter(catBPP(i,:),abs(catPSNR(i,:)),'o')

    % Plot the PSNR vs BPP for Chair.
    figure(f4);
    set(f4,'visible','off')
    subplot(2,4,i);
    stairs(chairBPP(i,:),abs(chairPSNR(i,:)),'LineWidth',1)
    title(sprintf('Feature Map #%i',i))
    xlabel('Bits Per Pixel');
    ylabel('PSNR (dB)')
    hold on;
    scatter(chairBPP(i,:),abs(chairPSNR(i,:)),'o')
end

% Save the image.
saveas(f1,'DataImages/Problem1/Boat/BoatStairs.jpg');
saveas(f2,'DataImages/Problem1/Bottle/BottleStairs.jpg');
saveas(f3,'DataImages/Problem1/Cat/CatStairs.jpg');
saveas(f4,'DataImages/Problem1/Chair/ChairStairs.jpg');
close all;
clear f1 f2 f3 f4;
fprintf('Stair Plots Created!\n');

% Display the Reconstructed Images.
j = 0:10:100;
for i = 1:8
    f5 = figure('Position',[300 300 900 500]);
    sgtitle({'Reconstructed Boat Images',sprintf('Feature Map %i',i)})
    f6 = figure('Position',[300 300 900 500]);
    sgtitle({'Reconstructed Bottle Images',sprintf('Feature Map %i',i)})
    f7 = figure('Position',[300 300 900 500]);
    sgtitle({'Reconstructed Cat Images',sprintf('Feature Map %i',i)})
    f8 = figure('Position',[300 300 900 500]);
    sgtitle({'Reconstructed Chair Images',sprintf('Feature Map %i',i)})
    imCt = 1;
    for k = 1:11
        % Plot the Reconstructed Feature Maps for Boat.
        figure(f5);
        set(f5,'visible','off')
        subplot(2,6,imCt);
        imagesc(boatRecon{i,k})
        title({sprintf('Quality: %i',j(k)),...
            sprintf('PSNR: %1.2f',abs(boatPSNR(i,k)))})
        
        % Plot the Reconstructed Feature Maps for Bottle.
        figure(f6);
        set(f6,'visible','off')
        subplot(2,6,imCt);
        imagesc(bottleRecon{i,k})
        title({sprintf('Quality: %i',j(k)),...
            sprintf('PSNR: %1.2f',abs(bottlePSNR(i,k)))})
        
        % Plot the Reconstructed Feature Maps for Cat.
        figure(f7);
        set(f7,'visible','off')
        subplot(2,6,imCt);
        imagesc(catRecon{i,k})
        title({sprintf('Quality: %i',j(k)),...
            sprintf('PSNR: %1.2f',abs(catPSNR(i,k)))})
        
        % Plot the Reconstructed Feature Maps for Chair.
        figure(f8);
        set(f8,'visible','off')
        subplot(2,6,imCt);
        imagesc(chairRecon{i,k})
        title({sprintf('Quality: %i',j(k)),...
            sprintf('PSNR: %1.2f',abs(chairPSNR(i,k)))})
        imCt = imCt + 1;
    end
    
    % Save the image.
    saveas(f5,strcat('DataImages/Problem1/Boat/BoatMap',num2str(i),'Recon.jpg'));
    saveas(f6,strcat('DataImages/Problem1/Bottle/BottleMap',num2str(i),'Recon.jpg'));
    saveas(f7,strcat('DataImages/Problem1/Cat/CatMap',num2str(i),'Recon.jpg'));
    saveas(f8,strcat('DataImages/Problem1/Chair/ChairMap',num2str(i),'Recon.jpg'));
    clearvars f5 f6 f7 f8;
    close all;
    fprintf(['Reconstructed Images of Feature Map ',...
        num2str(i),' Created!\n']);
end

return;


