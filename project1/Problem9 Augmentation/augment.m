load digits.mat;
numImage = size(X, 1);
totImage = 2 * numImage;
for i = numImage+1:totImage
    % randomly choose a image to augment
    sample = randi(numImage);
    img = reshape(X(sample, :), 16, 16);
    label = y(sample);

    % randomly choose an augmentation method
    type = randi(3);
    switch type
    case 1  % translation
        movement = randi([-2,2],[1,2]);  % max movement set to 2
        newImg = imtranslate(img, movement);
    case 2  % rotation
        angle = randi([-10,10]);
        newImg = imrotate(img, angle, 'bilinear', 'crop');
    case 3  % resize
        padding = randi(3);
        newSize = 16 - 2 * [padding, padding];
        resizeImg = imresize(img, newSize);
        newImg = padarray(resizeImg, [padding, padding]);
    case 4  % add Gaussian noise
        newImg = img + 30 * randn(16);
    end
    X(i,:) = newImg(:);
    y(i) = label;
end

save(['augmentedDigits.mat'], 'X', 'y', 'Xvalid', 'yvalid', 'Xtest', 'ytest')