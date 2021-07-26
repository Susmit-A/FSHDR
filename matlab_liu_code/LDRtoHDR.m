function out = LDRtoHDR(input, expo)
global gamma;

input = Clamp(input, 0, 1);
out = (input).^gamma;
out = out ./ expo;