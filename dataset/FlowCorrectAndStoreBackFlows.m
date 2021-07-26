function FlowCorrectAndStoreBackFlows(dataset)
    % clearvars; clc;
    global gamma;
    gamma = 2.2;
    %global CRF;
    %load BaslerCRF BaslerCRF;
    %CRF = BaslerCRF;
    addpath('../matlab_liu_code');

    train_path = fullfile(dataset, 'train'); 
    S = dir(fullfile(train_path, '*'));
    N = setdiff({S([S.isdir]).name},{'.','..'});
    for ii = 1:numel(N)
        curr_folder = N{ii};

        img = Clamp(im2single(imread(fullfile(train_path, curr_folder, 'dynamic/le.tif'))), 0, 1);
        imgs{1} = img(:, :, 1:3);
        img = Clamp(im2single(imread(fullfile(train_path, curr_folder, 'dynamic/me.tif'))), 0, 1);
        imgs{2} = img(:, :, 1:3);
        img = Clamp(im2single(imread(fullfile(train_path, curr_folder, 'dynamic/he.tif'))), 0, 1);
        imgs{3} = img(:, :, 1:3);

        fid = fopen(fullfile(train_path, curr_folder, 'input_exp.txt'));
        curExpo = 2.^cell2mat(textscan(fid, '%f'));
        fclose(fid);

        curInLDR = ComputeOpticalFlow(imgs, curExpo);
        nanInds1 = isnan(curInLDR{1});
        curInLDR{1}(nanInds1) = LDRtoLDR(curInLDR{2}(nanInds1), curExpo(2), curExpo(1));
        nanInds2 = isnan(curInLDR{3});
        curInLDR{3}(nanInds2) = LDRtoLDR(curInLDR{2}(nanInds2), curExpo(2), curExpo(3));
        for i = 1:3
            saveLDRs{i} = uint16(curInLDR{i}*2^16);
        end

        flow = ComputeOpticalFlow_reverse(imgs, curExpo);
        save(fullfile(train_path, curr_folder, 'rev_flow.mat'), 'flow');

        mkdir(fullfile(train_path, curr_folder, 'liu_flow_corrected'))
        imwrite(saveLDRs{1}, fullfile(train_path, curr_folder, 'liu_flow_corrected/le.tif'));
        imwrite(saveLDRs{2}, fullfile(train_path, curr_folder, 'liu_flow_corrected/me.tif'));
        imwrite(saveLDRs{3}, fullfile(train_path, curr_folder, 'liu_flow_corrected/he.tif'));
        disp(curr_folder);
    end

    val_path = fullfile(dataset, 'val'); 
    S = dir(fullfile(val_path, '*'));
    N = setdiff({S([S.isdir]).name},{'.','..'});
    for ii = 1:numel(N)
        curr_folder = N{ii};

        img = Clamp(im2single(imread(fullfile(val_path, curr_folder, 'dynamic/le.tif'))), 0, 1);
        imgs{1} = img(:, :, 1:3);
        img = Clamp(im2single(imread(fullfile(val_path, curr_folder, 'dynamic/me.tif'))), 0, 1);
        imgs{2} = img(:, :, 1:3);
        img = Clamp(im2single(imread(fullfile(val_path, curr_folder, 'dynamic/he.tif'))), 0, 1);
        imgs{3} = img(:, :, 1:3);

        fid = fopen(fullfile(val_path, curr_folder, 'input_exp.txt'));
        curExpo = 2.^cell2mat(textscan(fid, '%f'));
        fclose(fid);

        curInLDR = ComputeOpticalFlow(imgs, curExpo);
        nanInds1 = isnan(curInLDR{1});
        curInLDR{1}(nanInds1) = LDRtoLDR(curInLDR{2}(nanInds1), curExpo(2), curExpo(1));
        nanInds2 = isnan(curInLDR{3});
        curInLDR{3}(nanInds2) = LDRtoLDR(curInLDR{2}(nanInds2), curExpo(2), curExpo(3));
        for i = 1:3
            saveLDRs{i} = uint16(curInLDR{i}*2^16);
        end
        
        mkdir(fullfile(val_path, curr_folder, 'liu_flow_corrected'))
        imwrite(saveLDRs{1}, fullfile(val_path, curr_folder, 'liu_flow_corrected/le.tif'));
        imwrite(saveLDRs{2}, fullfile(val_path, curr_folder, 'liu_flow_corrected/me.tif'));
        imwrite(saveLDRs{3}, fullfile(val_path, curr_folder, 'liu_flow_corrected/he.tif'));
        disp(curr_folder);
    end
