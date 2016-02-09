function [newTraining rows] = modTrain(trainingPac,outputs,thres)
%outputs = result from training set    
outputs = outputs > thres;
    rows=[];
    for i =1 : size(trainingPac,1)
       if(trainingPac(i,6)==0 || trainingPac(i,6)==1)
           if(trainingPac(i,6)~=outputs(i))
               rows=[rows; i];
           end
       end
    end
    newTraining = removerows(trainingPac,'ind',rows);
    

end