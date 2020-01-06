from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import random

if __name__ == "__main__":
    # im = Image.new("RGB", (512, 512), "red")
    # im.show()
    #
    # testfile = "dot-grid-3x3_768x768.png"
    # im = Image.open(testfile)
    # print(type(im))
    # # JPEG (512, 512) RGB
    # im.save("lena.bmp")
    # im.show()
    # small = im.copy()
    # thsize = (128, 128)
    # small.thumbnail(thsize)
    # small.show()
    # box = (100, 100, 400, 400)
    # region = im.crop(box)
    # print("region", region.format, region.size, region.mode)
    # # region = region.transpose(Image.ROTATE_180)
    # region = region.transpose(Image.ROTATE_180)
    # region.show()
    # im.paste(region, box)
    # im.show()

    testfile = "Template1.jpg"
    # im = cv2.imread(testfile)
    #
    # new_im = Image.new("RGB", (512, 512), "black")
    #
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im_pil = Image.fromarray(im)
    #
    # im_pil.show()

    font = ImageFont.truetype("arial.ttf", 50)
    # font.size = 50
    print(font)
    # im = Image.open(testfile)
    im = Image.new("RGB", (2048, 2048), "white")
    draw = ImageDraw.Draw(im)
    text = "1234567ABCDEFJKLMPRTUVYaefhirt#&!?ΓΔλΨΩΣ¥£Π"
    draw.text((249, 455), text, font=font, fill=(0, 0, 0))
    print(font.getsize(text))
    # mask = ImageFont.getmask(text, font)
    # in PIL this code is written differently:
    # print(font.getsize(text))
    # mask = font.getmask(text)
    # print(type(mask))
    # cv2.imshow("mask", mask)
    im.show()