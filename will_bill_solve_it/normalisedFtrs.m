function normalisedFtrs(trainingPac, test)
labels = trainingPac(:,end);
bigMat = [trainingPac(:,1:end-1);test];    
for i =1:size(test,2)
       bigMat(:,i) = normalize(bigMat(:,i));    
end
trainingPac = bigMat(1:size(trainingPac,1),:);
test = bigMat(size(trainingPac,1)+1,:);
[ net ] = myTrain(trainingPac',labels',8 );
[z ySec] = myTest(net,test,[]);


end