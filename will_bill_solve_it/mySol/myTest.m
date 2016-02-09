function [y ySec] = myTest(net,TestMat,secondnet)
    y = net(TestMat');
    z= y>0.5;
    ySec=[];
%     mat=[];
%     ySec = secondnet(TestMat');
%     for i =1:size(TestMat,1)
%         mat=[mat ySec(i)];
%           if(ySec(i)<0.05 && y(i)>0.47 && y(i)<0.53)
%               z(i)=1-z(i);
%           end
%     end
%     
    %z = [(1:size(TestMat,1))' z'];
    z = [(0:size(TestMat,1)-1)' z'];
    headers = {'Id','solved_status'};
    csvwrite_with_headers('outSmash.csv',z,headers)
    %csvwrite('outSmash.csv',z);
end