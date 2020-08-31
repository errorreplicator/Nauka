import PIL as pil
from PIL import Image, ImageDraw


im = Image.new('RGB',[500,500],(255,0,255))
imDraw = ImageDraw.Draw(im)

imDraw.rectangle([(10,10),(210,110)],fill=(0,0,255),outline=(0,0,0))

im.show()



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
