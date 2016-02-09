function [input] = normalize(input)
    input = input-mean(input(:));
    input = input/std(input(:));
end