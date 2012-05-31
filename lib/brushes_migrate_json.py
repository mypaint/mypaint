
import sys, os
import brush

def migrate_brushes_to_json(dirpath):

    files = os.listdir(dirpath)
    files = [os.path.join(dirpath, fn)for fn in files if os.path.splitext(fn)[1] == '.myb']

    for fpath in files:
        b = brush.BrushInfo(open(fpath, 'r').read())
        open(fpath, 'w').write(b.to_json())

if __name__ == '__main__':

    directories = sys.argv[1:]
    for dir in directories:
        migrate_brushes_to_json(dir)
