function flow = ComputeCeLiu(target, source)
%%% We use Ce Liu's optical flow algorithm to compute the local flows. All the parameters are the default parameters.
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;
para = [alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations];

[vx, vy, ~] = Coarse2FineTwoFrames(target, source, para);
flow = single(cat(3, vx, vy));