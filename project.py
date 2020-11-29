import cv2 
import os
import numpy as np
import random as rand

PINK = (147,20,255)     # Pink BRG Color
SIZEY = 10              # 0.5x tracking box height
SIZEX = 6              # 0.5x tracking box width
RADIUS = 5             # Local exhaustive search radius


def findRed(img):
    '''
    This function takes an input image, detects the pixels containing red, and returns the image replacing the red
    pixels with blank ones.
    '''

    # converting from BGR to HSV color space
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Range for lower red
    lower_red = np.array([0,40,40])
    upper_red = np.array([25,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

    # Generating the final mask to detect red color
    mask1 = mask1+mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

    #creating an inverted mask to segment out the cloth from the frame
    mask2 = cv2.bitwise_not(mask1)

    #Segmenting the cloth out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(img,img,mask=mask2)

    return res1
   
def getWindow(img,row,col,n):
    '''
    This function takes an image and a pixel location and creates an nxn matrix of the window around the pixel
    '''

    size = int(n/2)
    out = np.zeros((n,n,3))

    for r in range(-size,size+1):
        for c in range(-size,size+1):
            out[r+size][c+size] = img[row+r][col+c]

    return out

def findMatches(T,I,size):
    '''

    '''

    rows,cols,trash = T.shape
    irows, icols, trash = I.shape


    # Create template validMask
    validMask = np.ones((rows,cols))
    for row in range(rows):
        for col in range(cols):
            if sum(T[row][col]) == 0:
                validMask[row][col] = 0

    # Create Gaussian kernel
    n = int((size-1)/2)
    sigma = size/6.4
    h1,h2 = np.meshgrid(np.linspace(-n,n,size),np.linspace(-n,n,size))
    kernel = np.exp(-(h1**2 + h2**2) / (2*sigma**2)) / (2*np.pi*sigma**2)


    # Calculate pixel costs
    ssd = np.zeros((irows,icols))
    TotWeight = np.sum(kernel*validMask)

    for i in range(n,irows-n):
        for j in range(n,icols-n):

            for ii in range(-n,n+1):
                for jj in range(-n,n+1):
                    cost = sum(T[ii+n,jj+n] - I[i+ii][j+jj])**2
                    ssd[i][j] += cost*validMask[ii+n,jj+n]*kernel[ii+n,jj+n]

    ssd /= TotWeight

    # Find low cost pixels (matches))
    ErrThresh = 0.2
    thresh = np.min(ssd[ssd>0])*(1+ErrThresh)

    PixelList = []
    for i in range(irows):
        for j in range(icols):
            if ssd[i][j] == 0:
                continue
            elif ssd[i][j] <= thresh:
                PixelList.append([i,j])
    
    return(PixelList)

def synthesize(red,img,sample,col,size):

    # Find red pixels in column
    rows = img.shape[0]
    record = []
    
    for row in range(70,125):
        
        # If red, preform synthesis
        if sum(red[row,col]) == 0:
            red[row,col] = [255,255,255]
            template = getWindow(red,row,col,size)
            matches = findMatches(template,sample,size)
            match = rand.choice(matches)
            img[row,col] = sample[match[0],match[1]]
            record.append([row,col,sample[match[0],match[1]]])


    return img,record

def drawBound(img,row,col):
    '''
    This function takes an image and positional arguments row and col and draws a box outline centered at 
    img[row][col]. The function outputs the original image with the tracking box drawn over it.
    '''
    for i in range(SIZEY+1):
        img[row+i][col+SIZEX] = PINK
        img[row-i][col+SIZEX] = PINK
        img[row+i][col-SIZEX] = PINK 
        img[row-i][col-SIZEX] = PINK

    for i in range(SIZEX+1):
        img[row+SIZEY][col+i] = PINK
        img[row+SIZEY][col-i] = PINK
        img[row-SIZEY][col+i] = PINK
        img[row-SIZEY][col-i] = PINK

    return img

def imageTracking(I,row,col,T,method):
    '''
    This function takes an image, I, and a template, T, as well as two positional arguments, row and col, and a 
    method argument for calculating cost. This function compares the image within the search radius of the positional
    argument to the template and calculates a cost using the method argument.

    The position with the lowest cost is saved and returned
    '''

    # Initialize cost and position
    cost = 99999999999999999999999
    pos = [row,col]

    # Extract tracking template
    subT = T[range(row-SIZEY,row+SIZEY+1),:]
    subT = subT[:,range(col-SIZEX,col + SIZEX+1)]

    # Search for nearest fit to template
    for r in range(-RADIUS,RADIUS+1):
        for c in range(-RADIUS,RADIUS+1):

            # Continue if attempting to search outside image
            if row+r-SIZEY < 0 or row+r+SIZEY+1 >= I.shape[0]:
                continue
            elif col+c-SIZEX < 0 or col + c + SIZEX+1 >= I.shape[1]:
                continue

            # Extract local search element from image
            subI = I[range(row+r-SIZEY,row+r+SIZEY+1),:]
            subI = subI[:,range(col+c-SIZEX,col + c + SIZEX+1)]

            # Calculate cost, keep if lower
            temp_cost = method(subI,subT)
            if temp_cost < cost:
                cost = temp_cost
                pos = [row+r,col+c]

    return pos

def ssd(I,T):
    '''
    This function takes an image, I, and a template, T, and returns the sum of squared difference 'cost'
    '''

    D = np.sum((I-T)**2)

    return D

# Read all images in folder
images = []
for filename in os.listdir('Input'):
    img = cv2.imread(os.path.join('Input',filename),cv2.IMREAD_COLOR)
    if img is not None:
        images.append(img)

# Find red pixels
img = images[0]


# Build sample image
sample = img[150:200,50:70]
record = []

##############################
## Run SSD
##############################

# Assign Boundary
pos = [30,284]   # Manually Assigned initial tracking position
outs = []       # Holds output images with boundary box

for i in range(len(images)-1):

    # Find position of pen
    pos = imageTracking(images[i+1],pos[0],pos[1],images[i],ssd)
    out = drawBound(images[i+1],pos[0],pos[1])
    out[40,pos[1]] = [255,255,255]

    
    # Replace appropriate pixels
    for pixel in record:
        out[pixel[0],pixel[1]] = pixel[2]

    red = findRed(out)

    for col in range(pos[1]-2,pos[1]+3):
        out,plus = synthesize(red,out,sample,col,9)
        record += plus

    outs.append(out)

# Create output video

frame = outs[0]
height, width, layers = frame.shape
video = cv2.VideoWriter('output.avi', 0, 20, (width,height))

for image in outs:
    video.write(image)

cv2.destroyAllWindows()
video.release()

# sample = img[150:200,50:70]


# rows, cols, trash = img.shape
# outs = []
# for col in range(1,cols):
#     red = findRed(img)
#     img = synthesize(red,sample,-col,9)
#     img[45,-col] = [255,255,255]
#     out = np.copy(img)
#     outs.append(out)


# # Create output video
# frame = outs[0]
# height, width, layers = frame.shape
# video = cv2.VideoWriter('output.avi', 0, 20, (width,height))

# for image in outs:
#     video.write(image)

# cv2.destroyAllWindows()
# video.release()




