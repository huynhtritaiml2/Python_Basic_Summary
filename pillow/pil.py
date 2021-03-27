from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

image = Image.open("hoian.jpg")
image2 = Image.open("dalat.jpg")

image.show() # Open image in photo viewer
#plt.imshow(image)
#plt.imshow(image2)

# Property of image
print(image.size) # (840, 460) # Width, Height ******************** in Numpy Height, Width ****************
print(image.format) # JPEG
print(image.mode) # RGB # RGB, BGR, HSV, Ycryb


# Save image 

image.save("newimage.jpg")

# crop ########
left = 50
top = 120
right = 250
bottom = 230
crop_image = image.crop((left,top,right,bottom))
#crop_image.show()
plt.imshow(crop_image)


#### copy a image
copied_image = image.copy()
#copied_image.show()


### Tranposing

transpose_image1 = image.transpose(Image.FLIP_LEFT_RIGHT)
transpose_image2 = image.transpose(Image.FLIP_TOP_BOTTOM)
transpose_image3 = image.transpose(Image.ROTATE_90)
transpose_image4 = image.transpose(Image.ROTATE_180)
transpose_image5 = image.transpose(Image.ROTATE_270)
transpose_image6 = image.transpose(Image.TRANSPOSE) # row to column, and ver via

plt.figure(figsize=(10,10))
plt.subplot(3, 2, 1)
plt.imshow(transpose_image1)

plt.figure(figsize=(10,10))
plt.subplot(3, 2, 2)
plt.imshow(transpose_image2)

plt.figure(figsize=(10,10))
plt.subplot(3, 2, 3)
plt.imshow(transpose_image3)

plt.figure(figsize=(10,10))
plt.subplot(3, 2, 4)
plt.imshow(transpose_image4)

plt.figure(figsize=(10,10))
plt.subplot(3, 2, 5)
plt.imshow(transpose_image5)

plt.figure(figsize=(10,10))
plt.subplot(3, 2, 6)
plt.imshow(transpose_image6)


### RESIZE
'''
NEAREST: nearest pixel from input image
BOX: 

Better we:

BICUBIC
LANCZOS

or deep learning

'''

newsize = (400,400)
plt.figure(figsize=(50,50))

resized_image1 = image.resize(newsize, Image.NEAREST)
resized_image2 = image.resize(newsize, Image.BILINEAR)
resized_image3 = image.resize(newsize, Image.BOX)
resized_image4 = image.resize(newsize, Image.HAMMING)
resized_image5 = image.resize(newsize, Image.BICUBIC)
resized_image6 = image.resize(newsize, Image.LANCZOS)


plt.subplot(3, 2, 1)
plt.imshow(resized_image1)
plt.title("Nearest")

plt.subplot(3, 2, 2)
plt.imshow(resized_image2)
plt.title("Bilinear")

plt.subplot(3, 2, 3)
plt.imshow(resized_image3)
plt.title("Box")

plt.subplot(3, 2, 4)
plt.imshow(resized_image4)
plt.title("Hamming")

plt.subplot(3, 2, 5)
plt.imshow(resized_image5)
plt.title("Bicubic")

plt.subplot(3, 2, 6)
plt.imshow(resized_image6)
plt.title("Lanczos")


######## Rotate

angle = 30
rotated_image = image.rotate(angle)
plt.imshow(rotated_image)

angle = -30
rotated_image = image.rotate(angle)
plt.imshow(rotated_image)


####### Text Watermark

from PIL import ImageFont
from PIL import ImageDraw

watermarked_image = image.copy() # No keep original image 
draw = ImageDraw.Draw(watermarked_image) # Draw object
font = ImageFont.truetype("arial.ttf", 100) # Choose font, download from internet # font object

# position, text, fill_color, font_object
draw.text((0, 0), "Sample Text", (0, 0, 0), font = font) # Black
draw.text((0, 0), "Sample Text", (255, 255, 255), font = font) # White

plt.imshow(watermarked_image)


################ Image watermark 

size = (500, 500)

crop_image = image.copy()
crop_image.thumbnail(size) # Preserves aspect ratio

copied_image = image.copy() # becuase we need to paste to original image
# Otiginal_image(thumnail, top_left position)
copied_image.paste(crop_image, (0, 0))
plt.imshow(copied_image)


## Convert to black and white image

bw_image = image.convert("L")
plt.imshow(bw_image, cmap='gray') # use cmap = gray for matplotlib to correctly show black and white


### COnvert to different Format
new_format_image = image.convert("HSV") # 
print(new_format_image.mode) # HSV


################## Convert to Numpy Format *********************
numpy_array = np.array(image)
print(numpy_array.shape) # (460, 840, 3) (Height, Width, 3) NOT PIL (Width, Height)

########## COnvert numpy array back to image
numpy_image = Image.fromarray(numpy_array)
plt.imshow(numpy_array)


######################3 Image Enhancement
'''
color enhancement: color enhance
sharpness enhancement: sharpness constract between dark and bright region are increased
constrast enhancement: enhance degree of color or grayscale variation are increased
brightness enhancement: overall image datkness is enhanced

Factor value = 1 (default)
1: no change
more than 1: increase
less than 1: decrease
'''

from PIL import ImageEnhance
plt.figure(figsize=(50,50))

image_color_enhan = image.copy()
image1 = ImageEnhance.Color(image_color_enhan).enhance(2.5)
image2 = ImageEnhance.Contrast(image_color_enhan).enhance(2.5)
image3 = ImageEnhance.Brightness(image_color_enhan).enhance(1.5)
image4 = ImageEnhance.Sharpness(image_color_enhan).enhance(2.5)


plt.subplot(2, 2, 1)
plt.imshow(image1)
plt.title("Color")

plt.subplot(2, 2, 2)
plt.imshow(image2)
plt.title("Constrast")

plt.subplot(2, 2, 3)
plt.imshow(image3)
plt.title("Brightness")

plt.subplot(2, 2, 4)
plt.imshow(image4)
plt.title("Sharpness")

#################### Alpha Blending
'''
out = image * (1.0 - alpha) + image2 * alpha
*png : have Alpha channel

alpha = 0 : copied of image 1
alpha = 1 : copied of image 2

'''
image = Image.open("hoian.jpg")
image2 = Image.open("dalat.jpg")

image1 = image.copy()
image2 = image2.copy()

image2 = image2.resize(image1.size)

image_blend = Image.blend(image1, image2, 0.5)
plt.imshow(image_blend)

########### Image Transform ************************ Difficult ***********************
'''
https://www.youtube.com/watch?v=dkp4wUhCwR4 : No source code
1. Affine Transform
2. Extent Transform
3. QUAD
4. MESH


Image_size , kind_transform, 6 Tupple value for Affine, or 4 Tupple for Extent
'''

image_transform = image.copy()
image_transform = image_transform.transform(image_transform.size, Image.AFFINE, (1, -0.5, 0.5 * image_transform.size[0],0,1,0))
plt.imshow(image_transform)



image_transform = image.copy()
image_transform = image_transform.transform(image_transform.size, Image.EXTENT, (100, 100, image_transform.size[0], image_transform.size[1]//2))
plt.imshow(image_transform)



##########################3 Flipping Channel *********************************
'''
RGB -> BGR 
'''

# RGB
image_channels = image.copy()
r, g, b = image_channels.split()
im = Image.merge("RGB", (r, g, b)) # RGB
plt.imshow(im)

# BGR
image_channels = image.copy()
r, g, b = image_channels.split()
im = Image.merge("RGB", (r, g, b)) # BGR *****************************
plt.imshow(im)