# Download ICCP19 Train Set  from https://www.kaggle.com/dataset/558d6f7da370e99824685b50488d9cb86fef812d31d68b9a64ec751b238978a6
# Download ICCP19 Test Set from https://www.kaggle.com/dataset/a9c5c05e9d5bf0de30009eb0714b461867c8e4a7ebc1288d705644827e27501f
# Then run this script with both train.zip and test.zip in this directory.

unzip test.zip
cat *.part* > test.tar
tar -xvf test.tar
python3 prepare_ICCP19.py --src Testing_set/ --dst ICCP19/val

rm -rf *.part*
unzip train.zip
cat *.part* > train.tar
tar -xvf train.tar
python3 prepare_ICCP19.py --src Training_set/ --dst ICCP19/train

rm -rf test.zip *.part*

matlab -nodisplay -nosplash -nodesktop -r "FlowCorrectAndStoreBackFlows('ICCP19'); exit;"