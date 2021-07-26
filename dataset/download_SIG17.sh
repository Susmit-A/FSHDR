# wget https://www.robots.ox.ac.uk/~szwu/storage/hdr/kalantari_dataset.zip
#unzip kalantari_dataset.zip
python prepare_SIG17.py
rm -rf train test
matlab -nodisplay -nosplash -nodesktop -r "FlowCorrectAndStoreBackFlows('SIG17'); exit;"
