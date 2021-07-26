function warped = ComputeOpticalFlow(imgs, expoTimes)
warped{2} = imgs{2};
v = ver;
havePar = any(strcmp('Parallel Computing Toolbox', {v.Name}));

expAdj{1} = AdjustExposure(imgs(1:2), expoTimes(1:2));
expAdj{2} = AdjustExposure(imgs(2:3), expoTimes(2:3));
expAdj{2} = expAdj{2}(2:-1:1);
flow = cell(1, 2);

if (havePar && ~isempty(gcp('nocreate')))
    parfor i = 1 : 2
        flow{i} = ComputeCeLiu(expAdj{i}{2}, expAdj{i}{1});
    end
else
    for i = 1 : 2
        flow{i} = ComputeCeLiu(expAdj{i}{2}, expAdj{i}{1});
    end
end
warped{1} = WarpUsingFlow(imgs{1}, flow{1});
warped{3} = WarpUsingFlow(imgs{3}, flow{2});