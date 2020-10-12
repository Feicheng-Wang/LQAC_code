import os
def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed, may already exist!" % path)
    else:
        print ("Successfully created the directory %s " % path)

# path = "../{}/log".format(FOLDER)

