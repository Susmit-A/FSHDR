import shutil, os, glob, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--src',   type=str, default='Testing_set')
parser.add_argument('--dst',   type=str, default='ICCP19/val')
args = parser.parse_args()

src = args.src
dst = args.dst
if not os.path.exists('ICCP19'):
    os.mkdir('ICCP19')
    os.mkdir(dst)
elif not os.path.exists(dst):
    os.mkdir(dst)
    
count = 1
for s in glob.glob(src + '/*'):
    if not os.path.exists(s + '/GT_HDR.hdr'):
        print(s + ' ignored due to no GT_HDR')
        continue
    
    d = dst + '/{:03d}/'.format(count)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.mkdir(d)
    os.mkdir(d + 'dynamic')
    os.mkdir(d + 'static')
    
    shutil.copyfile(s + '/GT_HDR.hdr', d + 'hdr_gt.hdr')
    if os.path.exists(s + '/ExpoBias.txt'):
        shutil.copyfile(s + '/ExpoBias.txt', d + 'input_exp.txt')
    elif os.path.exists(s + '/reference/ExposureBias.txt'):
        shutil.copyfile(s + '/reference/ExposureBias.txt', d + 'input_exp.txt')
    else:
        print('Both exp_bias variants unavailable in folder ' + s)
        continue
        
    if os.path.exists(s + '/ghosted/input_ghosted_1.tif'):
        shutil.copyfile(s + '/ghosted/input_ghosted_1.tif', d + '/dynamic/le.tif')
        shutil.copyfile(s + '/ghosted/input_ghosted_2.tif', d + '/dynamic/me.tif')
        shutil.copyfile(s + '/ghosted/input_ghosted_3.tif', d + '/dynamic/he.tif')
    elif os.path.exists(s + '/ghosted/img1.tif'):
        shutil.copyfile(s + '/ghosted/img1.tif', d + '/dynamic/le.tif')
        shutil.copyfile(s + '/ghosted/img2.tif', d + '/dynamic/me.tif')
        shutil.copyfile(s + '/ghosted/img3.tif', d + '/dynamic/he.tif')
    else:
        print('Both ghosted variants unavailable in folder ' + s)
        continue
    
    if os.path.exists(s + '/reference/input_reference_1.tif'):
        shutil.copyfile(s + '/reference/input_reference_1.tif', d + '/static/le.tif')
        shutil.copyfile(s + '/reference/input_reference_2.tif', d + '/static/me.tif')
        shutil.copyfile(s + '/reference/input_reference_3.tif', d + '/static/he.tif')
    else:
        paths = sorted(glob.glob(s + '/reference/EW4A*'))
        #print(paths[0], paths[1], paths[2])
        shutil.copyfile(paths[0], d + '/static/le.tif')
        shutil.copyfile(paths[1], d + '/static/me.tif')
        shutil.copyfile(paths[2], d + '/static/he.tif')
    count += 1
