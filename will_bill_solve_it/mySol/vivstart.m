training = load('../input.csv');
test = load('../../test/testAlt.csv');

sum(training(:,7)==0)
sum(training(:,7)==1)
sum(training(:,7)==0.5)
trainingNew = training;
zeroTrain = training(find((training(:,5)==0)));
zeroTrain = training(find((training(:,5)==0)),:);
trainingNew = [training ;zeroTrain];

sum(trainingNew(:,7)==0)
sum(trainingNew(:,7)==1)
sum(trainingNew(:,7)==0.5)

sum(trainingNew(:,8)==0)
sum(trainingNew(:,8)==1)
sum(trainingNew(:,8)==0.5)




