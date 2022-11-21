load digits.mat
[n,d] = size(X);
nLabels = max(y);
% yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [10];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;

% dropout probability for each layer, including input and hidden
% Following the original paper, 0.2 for input, 0.5 for hidden and no dropout for output layer, 
pDropout = [0.2, 0.5];

funObj = @(w,i)MLPSoftmaxLossWithDropout(w,X(i,:),y(i,:),nHidden,nLabels,pDropout);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredictWithDropout(w,Xvalid,nHidden,nLabels,pDropout);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
end

% Evaluate test error
yhat = MLPclassificationPredictWithDropout(w,Xtest,nHidden,nLabels,pDropout);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);