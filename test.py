# importing image object from PIL
import math
from PIL import Image, ImageDraw
import numpy as np

def calc_x2y2(length, radian, x1, y1):
    # (x2 = x1 + L.cos(θ),y2 = y1 + L.sin(θ))
    x2 = x1 + math.cos(radian)*length
    y2 = y1 + math.sin(radian)*length
    return (x2,y2)

w, h = 640, 640
x1,y1 = w/2, h/2
angle = 90 / 180 * np.pi
shape = [(x1,y1), calc_x2y2(300, angle , x1, y1)]
# creating new Image object
img = Image.new("RGB", (w, h),(225,225,225))
# create line image
img1 = ImageDraw.Draw(img)
img1.line(shape, fill="#401eba", width=5)
img.show()



# for x in range(3000):
#     if x<=255:
#         im = Image.new('RGB',[500,500],(x,0,0))
#     elif x>255 and x <= 510:
#         im = Image.new('RGB',[500,500],(255,x-255,0))
#     elif x>510 and x <=765:
#         im = Image.new('RGB',[500,500],(255,255,x-765))
#     else:
#         im = Image.new('RGB',[500,500],(255,255,255))
#     im.save(f'c:/1/{x}.png','png')

