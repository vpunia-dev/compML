function [net y] = myFtrsMod(input,testInput,secondnet,temp)
%gInput = [input(:,1)./(input(:,2)+1) input(:,3) input(:,4) input(:,5) input(:,6)./(input(:,6) + input(:,7)+1) input(:,8)/5 ];
%gInput = [input(:,1)./(input(:,2)+1) input(:,4) input(:,5) input(:,6)./(input(:,6) + input(:,7)+1) input(:,8)/5];

%gInput = [input(:,1) input(:,2) input(:,4)  input(:,5) input(:,6)];
gInput=input(:,1:end-1);
% [outputMatrix, meanInputVec ,stdInputVec] = normalizeMatrix(gInput');
% gInput = outputMatrix';

[ net ] = myTrain(gInput(:,1:end)',input(:,end)',4 );

%gtestInput = [testInput(:,27)./(testInput(:,28)+1) testInput(:,30) testInput(:,31) testInput(:,32)./(testInput(:,32) + testInput(:,33)+1) testInput(:,34)/5  ];
%gtestInput = [testInput(:,27)./(testInput(:,28)+1) testInput(:,29) testInput(:,30) testInput(:,31) testInput(:,32)./(testInput(:,32) + testInput(:,33)+1) testInput(:,34)/5  ];
%gtestInput = [testInput(:,27) testInput(:,28), testInput(:,29) testInput(:,30) testInput(:,31) testInput(:,32) testInput(:,33) testInput(:,34)];
% for i = 1:size(gtestInput,1)
%    gtestInput(i,:) = (gtestInput(i,:)-meanInputVec')./stdInputVec';
% end
gtestInput = testInput;
y = myTest(net,gtestInput,secondnet);
%y=myNewTest(net,gtestInput,temp);
end

function [outputMatrix, meanVec ,stdVec] = normalizeMatrix(inputMatrix)
        % All Vecs are column vectors
        outputMatrix=zeros(size(inputMatrix));
        if(size(inputMatrix,2)>1)
            meanVec = (mean(inputMatrix'))';
            stdVec = std(inputMatrix')'; % a column vector
            for i = 1: size(inputMatrix,2)
                outputMatrix(:,i)=inputMatrix(:,i)-meanVec;
                outputMatrix(:,i)=outputMatrix(:,i)./stdVec;
            end
        else
            % degenrate case
            outputMatrix=inputMatrix;
            meanVec=inputMatrix;
            stdVec=meanVec-meanVec;
        end
    end

