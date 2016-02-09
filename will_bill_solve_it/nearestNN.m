function [mdl, label] = nearestNN(training,testingbh)
mdl = fitcknn(training(:,1:10),training(:,11),'NumNeighbors',10);
label = predict(mdl,testingbh);
testNearest(label,0.49)
end