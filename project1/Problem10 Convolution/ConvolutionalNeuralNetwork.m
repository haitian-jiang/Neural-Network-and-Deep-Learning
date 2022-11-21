load digits.mat
[n,d] = size(X);
nLabels = max(y);
% yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xtest = standardizeCols(Xtest,mu,sigma);

% Choose network structure
nHidden = [10];
kernSize = 5;
d = (sqrt(d)-kernSize+1)^2;  % dimension after feature map

% Count number of parameters and initialize weights 'w'
nParams = kernSize^2 + d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)CNNSoftmaxLoss(w,X(i,:),y(i,:),nHidden,nLabels,kernSize);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = CNNclassificationPredict(w,Xvalid,nHidden,nLabels,kernSize);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
end

% Evaluate test error
yhat = CNNclassificationPredict(w,Xtest,nHidden,nLabels,kernSize);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);