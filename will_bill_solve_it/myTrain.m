function [ net ] = myTrain( X,L,no_nodes )
      net = feedforwardnet([no_nodes],'trainlm');
           % net = patternnet([no_nodes],'trainlm');
      setdemorandstream(34163765);
            %setdemorandstream(34163987);
%net = feedforwardnet([no_nodes],'trainscg');
    %net = patternnet([no_nodes],'trainlm');
    net.layers{2}.transferFcn='tansig';
    net.trainParam.epochs = 3000;
    [net tr] = train(net,X,L);
end

