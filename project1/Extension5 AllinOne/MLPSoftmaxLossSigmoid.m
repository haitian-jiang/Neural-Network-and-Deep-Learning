function [f,g] = MLPSoftmaxLossReLU(w,X,y,nHidden,nLabels,lambda)
Softmax = @(y) exp(y) ./ sum(exp(y));
sigmoid = @(x) 1./(1+exp(-x));
sigGrad = @(x) sigmoid(x).*sigmoid(-x);

[nInstances,nVars] = size(X);
% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1: offset + (nHidden(h-1)+1) * nHidden(h)), nHidden(h-1)+1, nHidden(h));
  offset = offset + (nHidden(h-1)+1) * nHidden(h);
end
% there is no need to distinguish output and hidden
hiddenWeights{length(nHidden)} = w(offset+1: offset + (nHidden(end)+1) * nLabels);
hiddenWeights{length(nHidden)} = reshape(hiddenWeights{end}, nHidden(end)+1 ,nLabels);

f = lambda * w'*w;
if nargout > 1
    gInput = 2*lambda * inputWeights;
    for h = 1:length(nHidden)
       gHidden{h} = 2*lambda * hiddenWeights{h}; 
    end
end

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = [1 sigmoid(ip{1})];
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = [1 sigmoid(ip{h})];
    end
    yhat = Softmax(fp{end}*hiddenWeights{end});
    label = y(i);
    f = f - log(yhat(label));
    
    if nargout > 1
        % derivative:
        % -log(yhat(l)) = -yhat(l) + log(sum(exp(yhat)))
        % d/dy log(sum(exp(yhat))) = yhat
        backprop = yhat;
        backprop(label) = backprop(label) - 1;  % -log(yhat(l))

        for h = length(nHidden):-1:1
            gHidden{h} = gHidden{h} + fp{h}'*backprop;
            % calculate backpropgation for previous layer
            backprop = (backprop*hiddenWeights{h}(2:end,:)').*sigGrad(ip{h});
        end

        % Input Weights
        gInput = gInput + X(i,:)'*backprop;

    end
    
end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1: offset + (nHidden(h-1)+1) * nHidden(h)) = gHidden{h-1};
        offset = offset + (nHidden(h-1)+1) * nHidden(h);
    end
    g(offset+1: offset + (nHidden(end)+1) * nLabels) = gHidden{end}(:);
end
