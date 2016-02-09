for i = 1:size(trainingPac,1)
       if(trainingPac(i,16)~=0 && trainingPac(i,16)~=1)
           trainingPac(i,16)= (trainingPac(i,2)+trainingPac(i,3))*(1.0/2.0);
           %trainingPac(i,7)= (trainingPac(i,2)+trainingPac(i,3)+1-trainingPac(i,6)/5)*(1.0/3.0);
       end
%     if(trainingPac(i,5)<=0.7)
%         trainingPac(i,5)= 0.5;
%     end
%     if(trainingPac(i,5)>=0.7)
%         trainingPac(i,5)= 1;
%     end
end