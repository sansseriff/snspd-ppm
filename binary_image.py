from PIL import Image
import numpy as np
with Image.open("C://Users//Andrew//Desktop//rover_l.png") as im:


    a = np.asarray(im)
    print(np.shape(a))

    print()np.shape
    np.save("binary_rover", a)