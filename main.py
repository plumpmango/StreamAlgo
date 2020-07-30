import stream as st
import sys
# import fichier as fic
# '../../../Video/1556887597.c035f88a39a20db55ad0d075babe6aae.mp4'

path = './video'
gradiant = 'sobel'
# fic.cleanFolder(path)
pathVideo = 0

if len(sys.argv) == 2:

    for arg in sys.argv:
        pathVideo = arg

if len(sys.argv) > 2:
    print("\tUsage : python main.py [videoNamePath]\n")
    print("NOTE : No parameter --> open webcam")
    exit()

st.streamVideo(pathVideo,path,gradiant)
