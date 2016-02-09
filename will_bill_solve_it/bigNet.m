function secondnet = bigNet(newTraining,temp)
    


    tempDatas = [temp(:,1:end-1);temp(:,1:end-1);temp(:,1:end-1)];
    
    trainData = [newTraining(:,1:end-1);tempDatas];
    zeroData = [zeros(size(temp,1),1);zeros(size(temp,1),1);zeros(size(temp,1),1)];
    
    testData = [newTraining(:,7)>0.75;zeroData];
    
    [ secondnet ] = myTrain(trainData',testData',4 );
    

end