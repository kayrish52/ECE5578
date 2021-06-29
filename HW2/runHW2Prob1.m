% This is the top level script for ECE5578 Homework 2: Problem 1.

% Clear the workspace.
clear all;
close all;

% Initialize the image file.
im = 'Lena.png';

% Read the image.
im = imread(im);

% Convert to Grayscale.
im = double(rgb2gray(im));

% Initialize the filter.
W = [0.333, 0.333, 0; 0.333, 0, 0; 0, 0, 0];

% Filter the image to a residual image.
imFilt = double(imfilter(im,W,'replicate'));

% Compute the residual.
imRes = imFilt-im;

% Initialize the centers of bins for histogram.
bounds = -255:255;
plotBounds = -30:30;

% Create the histogram.
imHistCounts = histcounts(imRes,bounds);
pImCounts = imHistCounts/sum(imHistCounts(:));

% Quantize the image and compare to the original.
q = 8;
[imQ,rMin,rMax,histQ,cBinsQ] = imQuant(im,imRes,imFilt,q);

% Generate the distribution from the Residual Distribution.
resPMF = histQ./sum(histQ(:));

% Sort the Probaility Distribution and the bins.
[resPDF,idx] = sort(resPMF,'descend');
cBinsQ = cBinsQ(idx);

% This function will generate the codebook map.
[codeBook,decodeBook,codeLength] = genCodeBook(cBinsQ);

% This will sync up the probability of the codeword with the length.
expLen = resPDF.*(codeLength');
expCodeLen = sum(expLen(:));

% This function will convert the symbols into a stream using a raster
% pattern.
imQStream = raster(imQ);

% This will generate a hardcoded bitstream.
bitStreamCodes = cell(length(imQStream),2);
totalLength = 0;
for i = 1:length(imQStream)
    bitStreamCodes{i,1} = codeBook(imQStream(i));
    bitStreamCodes{i,2} = length(bitStreamCodes{i,1});
    totalLength = totalLength + length(bitStreamCodes{i,1});
end

% This loop is responsible for the Context-Adaptive Binary Arithmetic
% Coding approach. It will perform multiple functions.
%  1. Initialize the parameters.
%  2. Concatenate 50 symbols and compute probabilities associated with
%     '0' or '1'.
%  3. Compute the binarized Arithmetic Coded stream of 50 symbols.
%  4. Repeat from step 2.
% 
% First, initialize the parameters.
statsStream = '';
encodedGroups = 1;
symLen = 50;
arGroups = ceil(totalLength/symLen);
encodedBlocks = cell(arGroups,1);
arCode = cell(arGroups,1);
fprintf('blocks needed: %5.2f\n',totalLength/symLen)
for i = 1:size(bitStreamCodes,1)

    % Sanity printout.
    if (mod(i,50000) == 0)
        fprintf('Iteration = %d\n',i);
    end
    
    % Capture the bitstreams for encoding and stat generation. Capture 50.
    statsStream = strcat(statsStream,bitStreamCodes{i,1});
    if (length(statsStream) < symLen)
        continue;
    end
    
    % Once 50 symbols are captured, store the bitstream.
    encodedBlocks{encodedGroups,1} = statsStream(1:symLen);
    
    % This function will calculate the arithmetic code, then convert it to
    % a binary bit stream.
    arCode{encodedGroups,1} = arCalc(statsStream(1:symLen));
    
    % Increment the group counter.
    encodedGroups = encodedGroups + 1;
    
    % Capture the remaining symbols that have not been encoded yet.
    statsStream = statsStream((symLen+1):end);
   
end

% Encode the last block.
if ~strcmp(statsStream,'')
    encodedBlocks{encodedGroups,1} = statsStream;
    arCode{encodedGroups,1} = arCalc(statsStream);
end

% This will calculate the average per pixel bit rate of the coding scheme.
totalBitsImQuant = 0;
totalBitsAr = 0;
for i = 1:arGroups
    totalBitsImQuant = totalBitsImQuant + length(encodedBlocks{i,1});
    totalBitsAr = totalBitsAr + length(arCode{i,1});
end

% Output the final bitrate.
imQBitRate = totalBitsImQuant/length(imQStream);
arBitRate = totalBitsAr/length(imQStream);

% Calculate the Entropy.
entropy = getEntropy(resPMF);

% Print the data on screen.
fprintf('For q = %d, the original per pixel bitrate is %1.3f\n',q,...
    imQBitRate)
fprintf('For q = %d, the arithmetic code per pixel bitrate is %1.3f\n',...
    q,arBitRate)
fprintf('For q = %d, the entropy is %1.3f\n',q,entropy)

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BELOW HERE ARE MANY OF THE SUPPORT FUNCTIONS FOR HW2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function will generate the codebook map to be used as the encoder.
function [codeBook,decodeBook,codeLength] = genCodeBook(bins)

% Determine the number of symbols.
l = length(bins);

% Determine the number of levels.
k = 0;
while (l > power(2,k))
    k = k + 1;
end

% Initialize the codebook arrays.
binaryCodes = cell(length(bins),1);
symbols = zeros(length(bins),1);
codeLength = zeros(length(bins),1);

% Encode the symbols into a codebook of binary representations.
symbolCount = 1;
for i = 0:k-1
    
    % Determine which level your exponential encoder is at. Set the range.
    % If the last bin, a special case exists.
    if i == 0
        range = 1;
        lastBin = 0;
    elseif i == k-1
        range = [(power(2,i)):l];
        lastBin = 1;
    else
        range = [(power(2,i)):(power(2,i+1)-1)];
        lastBin = 0;
    end
    
    % Loop over each of the symbols. Call the FUNCTION 'expGolombEncoder'
    % to generate the binary bitstream.
    for j = 1:length(range)
        symbols(symbolCount,1) = bins(range(j));
        binaryCodes{symbolCount,1} = expGolombEncoder(i,j-1,lastBin);
        codeLength(symbolCount,1) = length(binaryCodes{symbolCount,1});
        symbolCount = symbolCount + 1;
    end
end

% Once the symbol/bitstream pairs have been generated, store them in a Map
% Container for encoding and decoding.
codeBook = containers.Map(symbols,binaryCodes);
decodeBook = containers.Map(binaryCodes,symbols);
end

% This function is the Exponential Golomb Encoder.
function [binaryCode] = expGolombEncoder(preLen,postVal,lastBin)

% Special Case: if the Preamble Length is '0', than the bitstream is a 
% single binary '0'. Process no further.
if preLen == 0
    binaryCode = '0';
    return;
end

% Set the preamble code. Typically, it is a stream of '1's followed by a
% single '0'.
preCode = '';
for i = 1:preLen
    preCode = strcat(preCode,'1');
end

% The last bin requires special code generation. It has a trailing '1'
% instead of a trailing '0'.
if ~lastBin
    preCode = strcat(preCode,'0');
else
    preCode(end) = '1';
end

% Convert the symbol to a binary stream. This is done by modulo division
% and capturing the remainder.
r = zeros(1,preLen);
for i = 1:preLen
    r(i) = mod(postVal,2);
    postVal = floor(postVal/2);
end

% Generate the binary stream from the results of the modulo division step.
postCode = '';
for i = length(r):-1:1
    postCode = strcat(postCode,num2str(r(i)));
end

clear r postVal i preLen lastBin

% Conatenate the preable code and the binary stream to arrive at the final
% symbol stream.
binaryCode = strcat(preCode,postCode);
end

% This is the function that will convert the image into a stream using the
% raster pattern.
function [imRaster] = raster(im)

% Determine the size of the image and the total pixel count.
[h,w] = size(im);
pixelCount = w*h;

% Initialize all parameters. This portion will create a series of
% row/column pairs that retrieve the pixel values using a raster pattern.
r = 1;
c = 1;
d = '';
rowStream = zeros(pixelCount,1);
colStream = zeros(pixelCount,1);
imRaster = zeros(pixelCount,1);

% This loop captures the current row/column pair, saves the pair and the
% symbol from the image. Then it updates the pixel location.
for i = 1:pixelCount
    rowStream(i,1) = r;
    colStream(i,1) = c;
    imRaster(i,1) = im(r,c);
    [r,c,d] = updatePixel(r,c,d,h,w);
end
end

% This function works in conjunction with the raster function above to
% receive the current row and column, direction, as well as the height and
% width of the image being encoded. It uses these parameters to determine
% the current state of the raster pattern and to update the appropriate
% pixel location.
function [r,c,d] = updatePixel(r,c,d,h,w)

% Initialize the state vector.
state = zeros(9,1);

% Currently 9 states exist based on row and column values. Determine the 
% correct state.
state(1,1) = and(r==1,c==1);
state(2,1) = and(r==1,and(c~=1,c~=w));
state(3,1) = and(r==1,c==w);
state(4,1) = and(r~=1,and(r~=h,c==1));
state(5,1) = and(r~=1,and(r~=h,and(c~=1,c~=w)));
state(6,1) = and(r~=1,and(r~=h,c==w));
state(7,1) = and(r==h,c==1);
state(8,1) = and(r==h,and(c~=1,c~=w));
state(9,1) = and(r==h,c==w);

% Capture the state.
idx = find(state == 1, 1);

% This switch will determine the appropriate row, column, and direction
% updates based on the state of the image. It is a multi-layered decision
% logic process where most states need to consider the direction as well
% prior to updating the pixels. The states cover a variety of scenarios,
% where the row/column values take on a range from 1 to height/width, and
% the pixel must update based on the these inputs.
switch (idx)
    case 1
        d = 'r'; r = 1; c = c + 1;
    case 2
        if strcmp(d,'r'), d = 'dl'; r = 2; c = c - 1;
        elseif strcmp(d,'ur'), d = 'r'; r = 1; c = c + 1;
        end
    case 3
        r = 2;
        if strcmp(d,'r'), d = 'dl'; c = c - 1;
        elseif strcmp(d,'ur'), d = 'd';
        end
    case 4
        if strcmp(d,'dl'), d = 'd'; r = r + 1;
        elseif strcmp(d,'d'), d = 'ur'; r = r - 1; c = c + 1;
        end
    case 5
        if strcmp(d,'dl'), r = r + 1; c = c - 1;
        elseif strcmp(d,'ur'), r = r - 1; c = c + 1;
        end
    case 6
        r = r + 1;
        if strcmp(d,'ur'), d = 'd';
        elseif strcmp(d,'d'), d = 'dl'; c = c - 1;
        end
    case 7
        c = c + 1;
        if strcmp(d,'d'), d = 'ur'; r = r - 1;
        elseif strcmp(d,'dl'), d = 'r';
        end
    case 8
        c = c + 1;
        if strcmp(d,'dl'), d = 'r';
        elseif strcmp(d,'r'), d = 'ur'; r = r - 1;
        end
    case 9
        d = ''; r = h; c = w;
end
end

% This function is for the binary arithmetic encoder.
function [arBinCode] = arCalc(encodeStream)

% Initialize the range and the initial probabilities.
pLow = double(0);
pHigh = double(1);

% Initial the probability.
prob = 0.5*ones(1,2);

% Initialize the '0'/'1' counter.
count = ones(1,2);

% Capture the length of the bit stream to be encoded.
l = length(encodeStream);

% Loop over each bit in the stream.
for i = 1:l

    % Initialize the midpoint of the binary range based on the
    % probabilities. Then adjust the upper or lower boundary based on the
    % value of the next bit.
    mid = double(pLow + prob(1,1)*(pHigh-pLow));
    if strcmp(encodeStream(i),'0')
        pHigh = mid;
        count(1,1) = count(1,1) + 1;

        % Calculate the initial probabilities of '0's and '1's.
        prob(1,1) = count(1,1)/(count(1,1)+count(1,2));
        prob(1,2) = count(1,2)/(count(1,1)+count(1,2));
    else
        pLow = mid;
        count(1,2) = count(1,2) + 1;

        % Calculate the initial probabilities of '0's and '1's.
        prob(1,1) = count(1,1)/(count(1,1)+count(1,2));
        prob(1,2) = count(1,2)/(count(1,1)+count(1,2));
    end
end

% Capture the final arithmetic decimal code for the bit stream as the
% midpoint between the final upper and lower boundaries.
tag = double(pLow + prob(1,1)*(pHigh-pLow));

% Compute p(x).
pX = pHigh-pLow;
% fprintf('pLow  = %1.75f\npHigh = %1.75f\ntag   = %1.75f\n',pLow,pHigh,tag)

% This will encode the tag using the probability of zeros and ones
% generated.
arBinCode = arEncode(tag,prob,pX);

clear l mid i;
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

% This starts the imQuant function.
function [imQ,rMin,rMax,hist3,cBins3] = imQuant(im,imRes,imFilt,q)
% Quantize the residual image.
if q == 0
    imQ = imRes;
    q1 = 1;
else
    imQ = fix(imRes/q)*q;
    q1 = q;
end

% Calculate the minimum and maximum quantization residuals.
rMin = min(imQ(:));
rMax = max(imQ(:));

% Display the minimum residual and the maximum residual to the user.
% fprintf('\nMinimum Residual = %d\nMaximum Residual = %d\n',rMin,rMax);

% Compute histogram data.
[hist1,~] = hist(im, [0:255]);
[hist2,~] = hist(imRes(:), [-255:255]);
[hist3,cBins3] = hist(imQ(:), [rMin:q1:rMax]);

% Normalize the histogram data.
h1 = hist1./sum(hist1(:));
h2 = hist2./sum(hist2(:));
h3 = hist3./sum(hist3(:));

% Compute the Engropy of each image.
imEntropy = getEntropy(h1);
imResEntropy = getEntropy(h2);
imQEntropy = getEntropy(h3);

% Create a figure and establish a colormap.
figure('Position',[100,100,900,600]);
colormap('gray')

% Plot the original image.
subplot(2,3,1)
imagesc(im)
title('Original Lena Image')

% Plot the filtered image.
subplot(2,3,4)
imagesc(imFilt)
title({'Filtered Lena Image',sprintf('PSNR = %1.2fdB',-psnr(im,imFilt))})

% Plot the reconstructed image from residual.
imRecon = imFilt+imRes;
subplot(2,3,2)
imagesc(imRecon)
title({'Reconstructed Lena Image',...
    sprintf('PSNR = %1.2fdB',-psnr(imRecon,im))})

% Plot the residual image.
subplot(2,3,5)
imagesc(imRes)
title({'Residual Lena Image'})

% Plot the reconstructed image from quantized residual.
imQRecon = imFilt+imQ;
subplot(2,3,3)
imagesc(imQRecon)
title({'Quantized Reconstructed Lena Image',...
    sprintf('q = %d, PSNR = %1.2fdB',q,-psnr(im,imQRecon))})

% Plot the quantized residual image.
subplot(2,3,6)
imagesc(imQ)
title({'Quantized Residual Lena Image',sprintf('q = %d',q)})

% Create a figure. Establish the colormap.
figure('Position',[100 100 900 700]);
colormap('gray');

% Plot the original image.
subplot(3,3,1)
imagesc(im);
title('Original Image');

% Plot the residual image.
subplot(3,3,2);
imagesc(imRes);
title('Residual Image');

% Plot the quantized residual image.
subplot(3,3,3);
imagesc(imQ);
title('Quantized Residual Image');

% Compute the histogram of each image.
subplot(3,3,4)
histogram(im, [0:5:255]);
xticks(0:50:250);
xticklabels({'0','50','100','150','200','250'});
xlabel('Histogram Bin Centers')
ylabel('Counts')
title({'Residual Distribution','Lena.png'})

subplot(3,3,5)
histogram(imRes(:), [-35:35]);
xticks(-30:10:30);
xticklabels({'-30','-20','-10','0','10','20','30'});
xlabel('Histogram Bin Centers')
ylabel('Counts')
title({'Residual Distribution','Lena.png'})

subplot(3,3,6)
histogram(imQ(:), [-30:q1:30]);
xticks(linspace(-28,28,8));
xticklabels({'-28','-20','-12','-4','4','12','20','28'});
xlabel('Histogram Bin Centers')
ylabel('Counts')
title({'Quantized Residual Distribution','Lena.png'})

% Plot the Entropy of the original image.
subplot(3,3,7);
plot(h1,'.-b');
xticks(linspace(0,250,6));
xticklabels({'0','50','100','150','200','250'});
hold on;
grid on;
title({'Original Image',sprintf('Entropy = %1.2f',imEntropy)});

% Plot the Entropy of the residual image.
subplot(3,3,8);
plot([-255:255],h2,'.-k');
xticks(linspace(-250,250,6));
xticklabels({'-250','-150','-50','50','150','250'});
hold on;
grid on;
title({'Residual Image',sprintf('Entropy = %1.2f',imResEntropy)});

% Plot the Entropy of the quantized residual image.
subplot(3,3,9);
plot(linspace(rMin,rMax,length(h3)),h3,'.-r');
xticks(linspace(-250,250,11));
xticklabels({'-250','-200','-150','-100','-50','0','50','100',...
    '150','200','250'});
hold on;
grid on;
title({'Quantized Residual Image',sprintf('q = %d, Entropy = %1.2f',...
    q,imQEntropy)});

close all;
end

% This starts the getEntropy function.
function [e] = getEntropy(pmf)

% The Probability Mass Function requires positive data. This isolates
% values that are greater than zero.
pmf = pmf(pmf>0);

% Compute the Entropy of the Probability Mass Function.
e = -1*sum(pmf.*log(pmf));
end
