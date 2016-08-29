import pickle
import numpy as np 
from PIL import Image

def load_image(img_path):
    img = Image.open(img_path)
    return img


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")


filelist = open("train_1.txt","r")
filewrite = open("train_list.txt","w")

lines = filelist.readlines()

for line in lines:
	path = line.split(" ")[0]
	img = load_image(path)
	img = pil_to_nparray(img)
	if len(img.shape) != 3:
		print path
	else:
		if img.shape[2] == 3:
			filewrite.write(line)
		else:
			print path

filewrite.flush()
filewrite.close()


