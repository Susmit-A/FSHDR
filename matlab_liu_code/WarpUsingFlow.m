function warped = WarpUsingFlow(imgs, flows, needBase)

if (~exist('needBase', 'var') || isempty(needBase))
    needBase = true;
end

flows = gather(flows);

[hi, wi, c, numImages] = size(imgs);
[hf, wf, ~, ~] = size(flows);

hd = (hi - hf) / 2; wd = (wi - wf) / 2;

warped = zeros(hf, wf, c, numImages, 'single');

[X, Y] = meshgrid(1:wf, 1:hf);
[Xi, Yi] = meshgrid(1-wd:wf+wd, 1-hd:hf+hd);

for j = 1 : numImages
    for i = 1 : c
        
        if (needBase)
            curX = X + flows(:, :, 1, j);
            curY = Y + flows(:, :, 2, j);
        else
            curX = flows(:, :, 1, j);
            curY = flows(:, :, 2, j);
        end
        
        warped(:, :, i, j) = interp2(Xi, Yi, imgs(:, :, i, j), curX, curY, 'cubic', nan);
    end
end

warped = Clamp(warped, 0, 1);

% warped(isnan(warped)) = 0;