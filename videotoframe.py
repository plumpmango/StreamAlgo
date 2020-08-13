import cv2 as cv
import os


def vtofs(pathVideo,dirPathSave):
    # Opens the Video file
    cap= cv.VideoCapture(pathVideo)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv.imwrite(dirPathSave+'image'+str(i)+'.jpg',frame)
        print('image'+str(i)+'.jpg\n')
        i+=1

    cap.release()
    cv.destroyAllWindows()

#Main de test du module fact
pathVideo = '/home/utilisateur/Bureau/StageCerema/essais/acquisition/Result/Velo1/Smartek-6CD1460209002020_7_29_15_49_46.avi'
dirPathSave = '/home/utilisateur/Bureau/StageCerema/essais/Traitement/stream/videoPieton1/'

vtofs(pathVideo, dirPathSave)
