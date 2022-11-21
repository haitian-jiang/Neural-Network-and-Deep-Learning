load augmentedDigits.mat
[n,d] = size(X);
nLabels = max(y);
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
nHidden = [100];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams + (nHidden(h-1)+1) * nHidden(h);
end
nParams = nParams + (nHidden(end)+1) * nLabels;
w = randn(nParams,1);

% momentum variables
beta = 0.9;
deltaW = zeros(size(w));  % w_{t} - w_{t-1}

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
batchSize = 10;
lambda = 0.01;
funObj = @(w,i)MLPSoftmaxLossReLU(w,X(i:i+batchSize,:),y(i:i+batchSize,:),nHidden,nLabels,lambda);

for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = MLPclassificationPredictReLU(w,Xvalid,nHidden,nLabels);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
        stepSize = stepSize * 0.8;
    end
    
    i = max(1,ceil(rand*n)-batchSize);
    [f,g] = funObj(w,i);
    deltaW = -stepSize*g + beta*deltaW;
    w = w + deltaW;
end

% Evaluate test error
yhat = MLPclassificationPredictReLU(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);