function out = HDRtoLDR_Sen(input, expo)
global CRF;
maxImVal = 2^16 - 1;

input = single(input);
input = input * expo;
[length, c] = size(CRF);
out = input;

for i = 1 : c
    out(:, :, i) = interp1(CRF(:, i), 0 : length-1, input(:, :, i), 'nearest', 'extrap');
end

out = out / maxImVal;
out = Clamp(out, 0, 1);