import cv2 as cv
import os


def fstov(dirPath,dirName):
    fps=10
    print('Create Video')
    liste=sorted(os.listdir(dirPath))
    frame=cv.imread(dirPath+liste[0])
    h,l,c=frame.shape
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(dirName+'.avi',fourcc, fps, (l,h))
    for i in range(len(liste)):
        frame=cv.imread(dirPath+liste[i])
        out.write(frame)
        print str(i) + "\n"
    print('Video OK')
    out.release()

#Main de test du module fact
dirPath = '/home/utilisateur/Bureau/StageCerema/essais/Traitement/stream/sobelMax/'
dirName = 'mouvement1'
fstov(dirPath,dirName)
