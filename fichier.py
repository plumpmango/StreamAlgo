import os

def cleanFolder(path):
    files = os.listdir(path)

    for i in range(0,len(files)):
        os.remove(path + '/' + files[i])
