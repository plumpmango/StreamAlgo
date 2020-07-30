import cv2 as cv
import numpy as np
import os
import fichier as fic

# TODO: difference px a px entre les deux frames
# in : frame1 type Mat
# in : frame2 type Mat
#out matDiff type Mat

def diff(frame1,frame2):
    return cv.subtract(frame1,frame2)

# TODO: multiplication px a px entre les deux matrices en entree
#in : mat1 type Mat
#in : mat2 type Mat
#out : result type Mat
def multPx(mat1,mat2):
    return abs(cv.multiply(mat1,mat2))

def gradiantSobel(mat):
    #[dx, dy] = cv.spatialGradient(mat)
    kernelX = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    kernelY = np.transpose(kernelX)
    dx = cv.filter2D(mat,-1,kernelX)
    dy = cv.filter2D(mat,-1,kernelY)
    return dx,dy

def roberts(mat) :
    kernelX = np.array([-1, 1])
    kernelY = np.transpose(kernelX)
    dx = cv.filter2D(mat,-1,kernelX)
    dy = cv.filter2D(mat,-1,kernelY)
    return dx , dy

def prewitt(mat):
    kernelX = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
    kernelY = np.transpose(kernelX)
    dx = cv.filter2D(mat,-1,kernelX)
    dy = cv.filter2D(mat,-1,kernelY)
    return dx,dy

def laplacian(mat):
    kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    laplace = cv.filter2D(mat,-1,kernel)
    return laplace

def gradiantOr(dx,dy):
    return cv.bitwise_or(dx,dy)
# TODO: corriger
def gradiantSqrt(dx,dy):
    # dx, dy = cv.spatialGradient(mat)
    # dx2  = cv.multiply(dx,dx)
    # dy2 = cv.multiply(dy,dy)
    dx2 = np.power(dx,2)
    dy2 = np.power(dy,2)
    result = np.sqrt(cv.add(dx2,dy2))
    return result.astype(np.uint8)

def gradiantMax(dx,dy):
    dst = cv.max(abs(dx),abs(dy))
    return dst

def gradiantSum(dx,dy):
    return abs(dx) + abs(dy)

# TODO: Calcul du contour des objets en mouvements par l algorithme stream sur 3 images successives
def applyStream(frame1,frame2,frame3,gradiant='laplacian',fusion='not'):
    diff1 = diff(frame1,frame2)
    diff2 = diff(frame2,frame3)
    l,h = frame1.shape
    gx1 = np.zeros([l,h])
    gy1 = np.zeros([l,h])
    gx2 = np.zeros([l,h])
    gy2 = np.zeros([l,h])

    #Compute gradiant
    if(gradiant == 'prewitt'):
        gx1, gy1 = prewitt(diff1)
        gx2, gy2 = prewitt(diff2)
    if(gradiant == 'sobel'):
        gx1, gy1 = gradiantSobel(diff1)
        gx2, gy2 = gradiantSobel(diff2)
    if(gradiant == 'roberts'):
        gx1, gy1 = roberts(diff1)
        gx2, gy2 = roberts(diff2)
    if(gradiant == 'laplacian'):
        gx1 = laplacian(diff1)
        gx2 = laplacian(diff2)

    result = np.zeros([l,h])

    #multiplication elmt by elmt after amplitude computation
    if(fusion == 'or'):
        result = multPx(gradiantOr(gx1,gy1),gradiantOr(gx2,gy2))
    if(fusion == 'max'):
        result = multPx(gradiantMax(gx1,gy1),gradiantMax(gx2,gy2))
    if(fusion == 'sqrt'):
        result = multPx(gradiantSqrt(gx1,gy1),gradiantSqrt(gx2,gy2))
    if(fusion == 'sum'):
        result = multPx(gradiantSum(gx1,gy1),gradiantSum(gx2,gy2))
    if(fusion == 'not'):
        result = multPx(gx1,gx2)

    #Normalisation des valeurs
    cv.normalize(result,result,0,255,cv.NORM_MINMAX)
    return result

def streamVideo(pathVideo,path,gradiant):

    cap = cv.VideoCapture(pathVideo)

    if(cap.isOpened() == False):
        print("Error to opening video stream\n")

    prewitt = 'prewitt'
    roberts = 'roberts'

    #sobel init
    fic.cleanFolder(path)
    pathResultOr = './'+ gradiant + 'Or'
    pathResultMax = './' + gradiant + 'Max'
    pathResultSqrt = './' + gradiant + 'Sqrt'
    pathResultSum=  './' + gradiant + 'Sum'
    fic.cleanFolder(pathResultOr)
    fic.cleanFolder(pathResultMax)
    fic.cleanFolder(pathResultSqrt)
    fic.cleanFolder(pathResultSum)

    #prewitt init
    pathResultPOr = './'+ prewitt + 'Or'
    pathResultPMax = './' + prewitt + 'Max'
    pathResultPSqrt = './' + prewitt + 'Sqrt'
    pathResultPSum=  './' + prewitt + 'Sum'
    fic.cleanFolder(pathResultPOr)
    fic.cleanFolder(pathResultPMax)
    fic.cleanFolder(pathResultPSqrt)
    fic.cleanFolder(pathResultPSum)

    #roberts init
    pathResultROr = './'+ roberts + 'Or'
    pathResultRMax = './' + roberts + 'Max'
    pathResultRSqrt = './' + roberts + 'Sqrt'
    pathResultRSum=  './' + roberts + 'Sum'
    fic.cleanFolder(pathResultROr)
    fic.cleanFolder(pathResultRMax)
    fic.cleanFolder(pathResultRSqrt)
    fic.cleanFolder(pathResultRSum)

    #laplacian
    pathL = './laplacien'
    fic.cleanFolder(pathL)

    nFrame = 0

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray,(3,3),0)
        #save frame
        file = path +'/image' + str(nFrame) + '.jpg'
        # print file

        cv.imwrite(file,gray,[cv.IMWRITE_JPEG_QUALITY,95])
        # Display the resulting frame
        cv.imshow('frame',gray)

        if(nFrame >= 2):
            frame1 = cv.imread(path+'/image' + str(nFrame-2) + '.jpg',cv.IMREAD_GRAYSCALE)
            frame2 = cv.imread(path+'/image' + str(nFrame-1) + '.jpg',cv.IMREAD_GRAYSCALE)

            #sobel
            streamOr = applyStream(frame1,frame2,gray,gradiant,'or')
            cv.imwrite(pathResultOr + '/image'+str(nFrame-2)+'.jpg',streamOr,[cv.IMWRITE_JPEG_QUALITY,95])

            streamMax = applyStream(frame1,frame2,gray,gradiant,'max')
            cv.imwrite(pathResultMax + '/image'+str(nFrame-2)+'.jpg',streamMax,[cv.IMWRITE_JPEG_QUALITY,95])

            streamSum = applyStream(frame1,frame2,gray,gradiant,'sum')
            cv.imwrite(pathResultSum + '/image'+str(nFrame-2)+'.jpg',streamSum,[cv.IMWRITE_JPEG_QUALITY,95])

            streamSqrt = applyStream(frame1,frame2,gray,gradiant,'sqrt')
            cv.imwrite(pathResultSqrt + '/image'+str(nFrame-2)+'.jpg',streamSqrt,[cv.IMWRITE_JPEG_QUALITY,95])
            cv.imshow('SobelSqrt', streamSqrt)

            #prewitt
            streamPOr = applyStream(frame1,frame2,gray,prewitt,'or')
            cv.imwrite(pathResultPOr + '/image'+str(nFrame-2)+'.jpg',streamPOr,[cv.IMWRITE_JPEG_QUALITY,95])

            streamPMax = applyStream(frame1,frame2,gray,prewitt,'max')
            cv.imwrite(pathResultPMax + '/image'+str(nFrame-2)+'.jpg',streamPMax,[cv.IMWRITE_JPEG_QUALITY,95])

            streamPSum = applyStream(frame1,frame2,gray,prewitt,'sum')
            cv.imwrite(pathResultPSum + '/image'+str(nFrame-2)+'.jpg',streamPSum,[cv.IMWRITE_JPEG_QUALITY,95])

            streamPSqrt = applyStream(frame1,frame2,gray,prewitt,'sqrt')
            cv.imwrite(pathResultPSqrt + '/image'+str(nFrame-2)+'.jpg',streamPSqrt,[cv.IMWRITE_JPEG_QUALITY,95])
            cv.imshow('PrewittSqrt', streamPSqrt)

            #roberts
            streamROr = applyStream(frame1,frame2,gray,roberts,'or')
            cv.imwrite(pathResultROr + '/image'+str(nFrame-2)+'.jpg',streamROr,[cv.IMWRITE_JPEG_QUALITY,95])

            streamRMax = applyStream(frame1,frame2,gray,roberts,'max')
            cv.imwrite(pathResultRMax + '/image'+str(nFrame-2)+'.jpg',streamRMax,[cv.IMWRITE_JPEG_QUALITY,95])

            streamRSum = applyStream(frame1,frame2,gray,roberts,'sum')
            cv.imwrite(pathResultRSum + '/image'+str(nFrame-2)+'.jpg',streamRSum,[cv.IMWRITE_JPEG_QUALITY,95])

            streamRSqrt = applyStream(frame1,frame2,gray,roberts,'sqrt')
            cv.imwrite(pathResultRSqrt + '/image'+str(nFrame-2)+'.jpg',streamRSqrt,[cv.IMWRITE_JPEG_QUALITY,95])
            cv.imshow('RobertsSqrt', streamRSqrt)

            #Laplacien
            streamL = applyStream(frame1,frame2,gray,'laplacian','not')
            cv.imwrite(pathL + '/image'+str(nFrame-2)+'.jpg',streamL,[cv.IMWRITE_JPEG_QUALITY,95])
            cv.imshow('laplacien', streamL)
        nFrame+=1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
