function y = myNewTest(net,TestMat,temp)
    y = net(TestMat');
    z= y>0.5;
    for i = 1: size(TestMat,1)
        [~,index] = ismember(TestMat(i,:),temp(:,1:end-1),'rows');
        if(index~=0)
            z(i) = temp(index,end);
        end  
        if(rem(i,1000)==1)
            disp(i);
        end
    end
    %z = [(1:size(TestMat,1))' z'];
    z = [(0:size(TestMat,1)-1)' z'];
    headers = {'Id','solved_status'};
    csvwrite_with_headers('outSmash.csv',z,headers)
    %csvwrite('outSmash.csv',z);
end