import shutil, os, glob
src = 'train'
dst = 'SIG17/train'
if not os.path.exists('SIG17'):
    os.mkdir('SIG17')
    os.mkdir(dst)
elif not os.path.exists(dst):
    os.mkdir(dst)

count = 1
for s in glob.glob(src + '/*'):
    d = dst + '/{:03d}'.format(count)
    if not os.path.exists(d):
        os.mkdir(d)
        os.mkdir(d + '/dynamic')
        os.mkdir(d + '/static')
        
    shutil.copyfile(s + '/input_1_aligned.tif', d + '/dynamic/le.tif')
    shutil.copyfile(s + '/input_2_aligned.tif', d + '/dynamic/me.tif')
    shutil.copyfile(s + '/input_3_aligned.tif', d + '/dynamic/he.tif')
    shutil.copyfile(s + '/ref_1_aligned.tif', d + '/static/le.tif')
    shutil.copyfile(s + '/ref_2_aligned.tif', d + '/static/me.tif')
    shutil.copyfile(s + '/ref_3_aligned.tif', d + '/static/he.tif')
    shutil.copyfile(s + '/ref_hdr_aligned.hdr', d + '/hdr_gt.hdr')
    shutil.copyfile(s + '/input_exp.txt', d + '/input_exp.txt')
    print(str(count) + ' folders transferred')
    count += 1
    

src = 'test'
dst = 'SIG17/val'
if not os.path.exists(dst):
    os.mkdir(dst)

count = 1
for s in glob.glob(src + '/*'):
    d = dst + '/{:03d}'.format(count)
    if not os.path.exists(d):
        os.mkdir(d)
        os.mkdir(d + '/dynamic')
        
    shutil.copyfile(s + '/input_1_aligned.tif', d + '/dynamic/le.tif')
    shutil.copyfile(s + '/input_2_aligned.tif', d + '/dynamic/me.tif')
    shutil.copyfile(s + '/input_3_aligned.tif', d + '/dynamic/he.tif')
    shutil.copyfile(s + '/ref_hdr_aligned.hdr', d + '/hdr_gt.hdr')
    shutil.copyfile(s + '/input_exp.txt', d + '/input_exp.txt')
    print(str(count) + ' folders transferred')
    count += 1