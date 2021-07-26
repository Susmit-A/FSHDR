function out = LDRtoHDR_Sen(input, expo)
global CRF;
maxImVal = 2^16-1;

[h, w, c] = size(input);
out = zeros(h*w, c);
input = single(input) * maxImVal;
for i = 1 : c
    out(:, i) = CRF(uint16(input(:, :, i))+1, i);
end
out = single(reshape(out, h, w, c) / expo);