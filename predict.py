from frcnn import FRCNN
from PIL import Image

frcnn = FRCNN()

while True:
    img_filename = input("Input image filename:")
    # img_filename = "img/street.jpg"
    try:
        image = Image.open(img_filename)
    except:
        print("Open error, Try again")
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
