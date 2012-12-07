function [out1, out2] = test_function(in1,in2)
% This is a dummy MATLAB function for testing shell usage.

I1 = eye(10);
I2 = eye(5);
out1 = in1*I1 + in2*I1;
out2 = in1*in2*I2;
disp('---')
disp(out1)
disp(out2)
quit; 
