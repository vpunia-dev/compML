function testNearest(label,threshold)
    z = label>threshold;
    %z = [(1:size(TestMat,1))' z'];
    z = [(0:size(label,1)-1)' z];
    headers = {'Id','solved_status'};
    csvwrite_with_headers('outSmash.csv',z,headers)
end