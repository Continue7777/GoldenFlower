from PIL import Image
import matplotlib.pyplot as plt # plt 用于显示图片
pic = Image.open('249382611.jpg')
plt.imshow(pic)
plt.show()

