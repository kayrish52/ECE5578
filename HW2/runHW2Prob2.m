% This is the top level script for ECE5578 Homework 2 - Problem 2.

% Clear the workspace.
clear all;
close all;

% Initialize the image file.
im = 'Lena.png';

% Read the image.
im = imread(im);

% Create the binary image.
im = biLevelIm(im);

% Compute the histogram data.
[histBIM,cBins] = hist(im(:), [0,255]);

% Generate the distribution from the Residual Distribution.
resPMF = histBIM./sum(histBIM(:));

% Sort the Probaility Distribution and the bins.
[resPDF,idx] = sort(resPMF,'descend');
cBins = cBins(idx);

% This function will generate the codebook map.
[codeBook,decodeBook,codeLength] = genCodeBook(cBins);

% This loop is responsible for the first part of the Context-Aware Binary 
% Arithmetic Coding Algorithm. It will perform multiple functions.
%  1. Initialize the parameters.
%  2. Split each row into bins. The number of bins is determined by the 
%     the image width/50 + 1.
%  3. For each row, concatenate 50 symbols per bin. For the last bin, 
%     concatenate the remaining symbols.
%  4. Store the full image as a cell array of bins for each row, and each
%     row split into the appropriate number of bins.
%  5. Continue this process until the entire image is stored as bitstream 
%     bins.
% 
% First, initialize the bitstream cell array.
symLen = 50;
widthBins = ceil((size(im,2)-1)/symLen);
imBitStreams = cell(size(im,1)-1,widthBins);

% Loop over each row to encode 50 symbol bins.
for i = 2:size(im,1)
    % Sanity Printout
    if mod(i,50) == 0
        fprintf('Iteration = %d\n',i);
    end
    
    % For each bin, convert the image into a stream of symbols.
    for j = 1:widthBins
        % Initialize the bitstream.
        imBitStream = '';
        
        % If the final bin, capture the remaining symbols. Else, capture 50
        % symbols.
        if j == widthBins
            bitStream = im(i,(((j-1)*symLen)+2):end);
            bitStream(bitStream == 255) = 1;
            for k = 1:length(bitStream)
                imBitStream = strcat(imBitStream,num2str(bitStream(k)));
            end
            imBitStreams{i-1,j} = imBitStream;
        else
            bitStream = im(i,(((j-1)*symLen)+2):((j*symLen)+1));
            bitStream(bitStream == 255) = 1;
            for k = 1:length(bitStream)
                imBitStream = strcat(imBitStream,num2str(bitStream(k)));
            end
            imBitStreams{i-1,j} = imBitStream;
        end
    end
end

% This loop is responsible for the second part of Context-Aware Binary 
% Arithmetic Coding Algorithm. It will perform multiple functions.
%  1. Initialize the parameters.
%  2. Capture the image row data to be analyzed for Context-Aware Binary 
%     Arithmetic Coding to generate the probabilities associated with each 
%     state. We are concerned with 3 surrounding pixels and 8 states.
%  3. Analyze the row data and compute the probabilities associated with 
%     each of the 8 states for the entire row.
%     per stream.
%  4. Encode each bin from part 1 using the probabilities computed in 
%     part 3 using a binary arithmetic coding technique.
%  5. Repeat from step 2.
% 
% First, initialize the parameters.
statsData = zeros(2,size(im,2));
encodeStream = '';
encodeCount = 1;
encodedGroups = 1;
arCode = cell(size(im,1)-1,widthBins);
for i = 2:size(im,1)
    
    % Encode each bin using the statistics data and the state previously
    % computed.
    for j = 1:widthBins
        if j == widthBins
            arCode{i-1,j} = biArEncode(imBitStreams{i-1,j},...
                im((i-1):i,((j-1)*symLen+1):end));
        else
            arCode{i-1,j} = biArEncode(imBitStreams{i-1,j},...
                im((i-1):i,((j-1)*symLen+1):((j*symLen)+1)));
        end
    end
end

% This will calculate the average per pixel bit rate of the coding scheme.
totalBits = 0;
totalArBits = 0;
for i = 1:size(imBitStreams,1)
    for j = 1:size(imBitStreams,2)
        totalBits = totalBits + length(imBitStreams{i,j});
        totalArBits = totalArBits + length(arCode{i,j});
    end
end

% Output the final bit rate.
bitRate = totalBits/(511*511);
arBitRate = totalArBits/(511*511);

% Output the Entropy.
entropy = getEntropy(resPMF);

% Print the data on screen.
fprintf('The binary image per pixel bitrate is %1.3f\n',bitRate)
fprintf(['The context aware arithmetic coded per pixel ', ...
    'bitrate is %1.3f\n'],arBitRate)
fprintf('The entropy is %1.3f\n',entropy)

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BELOW HERE ARE MANY OF THE SUPPORT FUNCTIONS FOR HW2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function will generate the codebook map to be used as the encoder.
function [codeBook,decodeBook,codeLength] = genCodeBook(bins)

% Initialize the codebook arrays.
binaryCodes = cell(length(bins),1);
symbols = zeros(length(bins),1);
codeLength = zeros(length(bins),1);

% For a binary image, the codewords are simply '0' or '1'.
binaryCodes{1,1} = '0';
binaryCodes{2,1} = '1';

% Identify the symbols.
symbols(1,1) = 0;
symbols(2,1) = 255;

% Associate the symbols with the codewords.
codeBook = containers.Map(symbols,binaryCodes);
decodeBook = containers.Map(binaryCodes,symbols);
end

% This function will analyze the state of the stream and generate a
% probability to the next symbol.
function [stats,stateVec] = stateAnalysis(imData)

imData = double(imData);
imData(imData == 255) = 1;

% Initialize each state of the statistics cell array, and the state vector.
stats = statsInit();
stateVec = zeros(1,size(imData,2));

% Loop over the image data, and adjust each probability based on the state
% of the surrounding pixel to the left, top-left, and top of the present
% pixel.
for i = 2:size(imData,2)
    
    % Extract the pixel data from the images.
    s1 = imData(2,i-1);
    s2 = imData(1,i-1);
    s3 = imData(1,i);

    % Initialize the state vector.
    state = zeros(8,1);

    % Determine the state of the pixel based on the surrounding pixels.
    state(1,1) = and(~s1,and(~s2,~s3));
    state(2,1) = and(s1,and(~s2,~s3));
    state(3,1) = and(~s1,and(s2,~s3));
    state(4,1) = and(s1,and(s2,~s3));
    state(5,1) = and(~s1,and(~s2,s3));
    state(6,1) = and(s1,and(~s2,s3));
    state(7,1) = and(~s1,and(s2,s3));
    state(8,1) = and(s1,and(s2,s3));
    
    % Capture the state.
    idx = find(state == 1);
    
    % Predict the value out of [0,1] for the current pixel based on the
    % state of the surrounding pixel. Adjust the appropriate count in the 
    % stats.
    if strcmp(num2str(imData(2,i)),'0')
        stats{idx,2} = stats{idx,2} + 1;
    else
        stats{idx,3} = stats{idx,3} + 1;
    end
    
    % Update the state vector to capture the state of the present pixel.
    stateVec(1,i) = idx;
end

% Calculate the probabilities for all 8 states.
for i = 1:8
    if or(stats{i,2} ~= 0,stats{i,3} ~= 0)
        stats{i,4} = stats{i,2}/(stats{i,2}+stats{i,3});
        stats{i,5} = stats{i,3}/(stats{i,2}+stats{i,3});
    else
        continue
    end
end
end


% This function is for the binary arithmetic encoder.
function [arBinCode] = biArEncode(encodeStream,statsData)

% Capture the statistics bitstream and compute the statistics as well
% as the state of each bit.
[stats,stateVec] = stateAnalysis(statsData);

% Initialize the range and the initial probabilities.
pLow = double(0);
pHigh = double(1);

% Initial the context streams.
cStreams = cell(8,1);
prob = zeros(8,2);

% Capture the probability set and store the probabilities in a map.
for i = 1:8
    cStreams{i,1} = stats{i,1};
    prob(i,1) = stats{i,4};
    prob(i,2) = stats{i,5};
end
p0Map = containers.Map(cStreams,prob(:,1));

% Capture the length of the bit stream to be encoded.
l = length(encodeStream);

% Loop over each bit in the binary image data.
for i = 1:l
    
    % Capture the correct probability from the statistics array based on
    % the state of the current pixel.
    p0 = p0Map(char(cStreams(stateVec(1,i+1))));
    
    % Initialize the midpoint of the binary range based on the
    % probabilities. Then adjust the upper or lower boundary based on the
    % value of the current pixel.
    mid = double(pLow + p0*(pHigh-pLow));
    if strcmp(encodeStream(i),'0')
        pHigh = mid;
    else
        pLow = mid;
    end
end

% Compute the final p0 as a weighted average of the different states
% observed in the bitstream.
stateVec = stateVec(1,2:end);
stateCounts = zeros(8,1);
for i = 1:8
    stateCounts(i,1) = length(find(stateVec == i));
end
stateWeights = stateCounts./sum(stateCounts(:));
p0 = stateWeights.*prob(:,1);
p0 = sum(p0.*stateCounts)/sum(stateCounts);

% Capture the final arithmetic decimal code for the bit stream as the
% midpoint between the final upper and lower boundaries.
tag = double(pLow + p0*(pHigh-pLow));

% Compute p(x).
pX = pHigh-pLow;
% fprintf('pLow  = %1.75f\npHigh = %1.75f\ntag   = %1.75f\n',pLow,pHigh,tag)

clear i l p0Map cStreams statsData stats stateVec

% This will encode the tag using the probability of zeros and ones
% generated.
arBinCode = arEncode(tag,[p0,(1-p0)],pX);

clear pX p0 stateCounts stateWeights pHigh pLow mid prob
end

% This function will encode the arithmetic values.
function [encodeStream] = arEncode(arCode,prob,pX)

% Determine the codelength based on the arithmetic code value.
codeLen = ceil(log2(1/pX)) + 1;

% Initialize the encoded arithmetic stream.
encodeStream = '';

% Set the high and low probabilities to use for encoding.
pLow = 0;
pHigh = 1;

% Loop over the arithmetic code to convert it to a binary stream.
for i = 1:codeLen
    
    % Establish the 'midpoint'.
    mid = pLow + prob(1,1)*(pHigh-pLow);
    
    % Encode the arithmetic value.
    if arCode >= mid
        encodeStream = strcat(encodeStream,'1');
        pLow = mid;
    else
        encodeStream = strcat(encodeStream,'0');
        pHigh = mid;
    end
end

clear mid i
end

% This starts the arithCode function.
function [bIm] = biLevelIm(im)

% Convert the Original Image to Grayscale.
im = rgb2gray(im);

% Binarize the original image.
bIm = im;
bIm(bIm < 128) = 0;
bIm(bIm >= 128) = 255;

% Create a figure. Establish the colormap.
figure('Position',[300,300,700,400]);
colormap('gray');

% Plot the original image.
subplot(1,2,1)
imagesc(im);
title('Original Image');

% Plot the residual image.
subplot(1,2,2);
imagesc(bIm);
title('Residual Image');

close all;
end

% This function simply exectes the initiation steps of the statistics
% array.
function [stats] = statsInit()

% Initialize the statistics cell array.
stats = cell(8,5);

% Initialize all fields of the statistics array.
stats{1,1}='000';   stats{1,2}=1;     stats{1,3}=1;
stats{1,4}=0.5;     stats{1,5}=0.5;

stats{2,1}='100';   stats{2,2}=1;     stats{2,3}=1;
stats{2,4}=0.5;     stats{2,5}=0.5;

stats{3,1}='010';   stats{3,2}=1;     stats{3,3}=1;
stats{3,4}=0.5;     stats{3,5}=0.5;

stats{4,1}='110';   stats{4,2}=1;     stats{4,3}=1;
stats{4,4}=0.5;     stats{4,5}=0.5;

stats{5,1}='001';   stats{5,2}=1;     stats{5,3}=1;
stats{5,4}=0.5;     stats{5,5}=0.5;

stats{6,1}='101';   stats{6,2}=1;     stats{6,3}=1;
stats{6,4}=0.5;     stats{6,5}=0.5;

stats{7,1}='011';   stats{7,2}=1;     stats{7,3}=1;
stats{7,4}=0.5;     stats{7,5}=0.5;

stats{8,1}='111';   stats{8,2}=1;     stats{8,3}=1;
stats{8,4}=0.5;     stats{8,5}=0.5;
end

% This starts the getEntropy function.
function [e] = getEntropy(pmf)

% The Probability Mass Function requires positive data. This isolates
% values that are greater than zero.
pmf = pmf(pmf>0);

% Compute the Entropy of the Probability Mass Function.
e = -1*sum(pmf.*log(pmf));
end
