function ApplyReverseFlow(dataset, model_name, save_location)
    clc;
    global gamma;
    gamma = 2.2;
    addpath('matlab_liu_code');

    train_path = fullfile('dataset', dataset, 'train');
    synthetic_HDR_path = fullfile('results', model_name, strcat('HDR_', dataset));
    if nargin > 2
        save_path = save_location   %Preferably at 'dataset/temp_datasets/<model_name>'
    else
        save_path = fullfile('dataset/temp_datasets', model_name);
    end
    mkdir(save_path)
    
    S = dir(fullfile(train_path, '*'));
    N = setdiff({S([S.isdir]).name},{'.','..'});
    disp(numel(N))
    for ii = 1:numel(N)
        curr_folder = N{ii};
        fid = fopen(fullfile(train_path, curr_folder, 'input_exp.txt'));
        curExpo = 2.^cell2mat(textscan(fid, '%f'));
        fclose(fid);
        
        f = load(fullfile(train_path, curr_folder, 'rev_flow.mat'));
        flow = f.flow;
        hdr_synthetic = hdrread(fullfile(synthetic_HDR_path, curr_folder, 'synthetic.hdr'));

        mod_HDR{1} = WarpUsingFlow(hdr_synthetic, flow{1});
        mod_HDR{2} = hdr_synthetic;
        mod_HDR{3} = WarpUsingFlow(hdr_synthetic, flow{2});
        
        curInLDR{1} = HDRtoLDR(mod_HDR{1}, curExpo(1));
        curInLDR{2} = HDRtoLDR(mod_HDR{2}, curExpo(2));
        curInLDR{3} = HDRtoLDR(mod_HDR{3}, curExpo(3));

        nanInds1 = isnan(curInLDR{1});
        curInLDR{1}(nanInds1) = LDRtoLDR(curInLDR{2}(nanInds1), curExpo(2), curExpo(1));
        nanInds2 = isnan(curInLDR{3});
        curInLDR{3}(nanInds2) = LDRtoLDR(curInLDR{2}(nanInds2), curExpo(2), curExpo(3));
        for i = 1:3
            saveLDRs{i} = uint16(curInLDR{i}*2^16);
        end
        
        mkdir(fullfile(save_path, curr_folder))
        imwrite(saveLDRs{1}, fullfile(save_path, curr_folder, 'le_synthetic.tif'));
        imwrite(saveLDRs{2}, fullfile(save_path, curr_folder, 'me_synthetic.tif'));
        imwrite(saveLDRs{3}, fullfile(save_path, curr_folder, 'he_synthetic.tif'));
        copyfile(fullfile(synthetic_HDR_path, curr_folder, 'synthetic.hdr'), fullfile(save_path, curr_folder, 'synthetic.hdr'))
        copyfile(fullfile(train_path, curr_folder, 'input_exp.txt'), fullfile(save_path, curr_folder, 'input_exp.txt'))
        disp(curr_folder);
    end