function [y] = CNNclassificationPredict(w,X,nHidden,nLabels,kernSize)
[nInstances,nVars] = size(X);
nVars = (sqrt(nVars)-kernSize+1)^2;

% Form Weights
kernel = reshape(w(1:kernSize^2), kernSize, kernSize);
offset = kernSize^2;
inputWeights = reshape(w(offset+1:offset+nVars*nHidden(1)),nVars,nHidden(1));
offset = offset + nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
for i = 1:nInstances
    img = reshape(X(i,:), 16, 16);
    feature = conv2(img, kernel, 'valid');
    featureVec = reshape(feature, 1, []);

    ip{1} = featureVec*inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    y(i,:) = fp{end}*outputWeights;
end
[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
