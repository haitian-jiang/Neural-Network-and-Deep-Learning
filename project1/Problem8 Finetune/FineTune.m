function w = FineTune(w,X,yExpanded,nHidden,nLabels)

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
% outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
% outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
IP{1} = X * inputWeights;
FP{1} = tanh(IP{1});
for h = 2:length(nHidden)
    IP{h} = FP{h-1}*hiddenWeights{h-1};
    FP{h} = tanh(IP{h});
end
outputWeights = ( (FP{end}' * FP{end}) \ FP{end}' ) * yExpanded;
w(offset+1:offset+nHidden(end)*nLabels) = outputWeights(:);