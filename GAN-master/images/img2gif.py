import os
from tqdm import tqdm
from PIL import Image

def img2gif():
    imgs = os.listdir(os.getcwd())
    imgs.sort()
    images = []
    total = len(imgs)
    for img in imgs:
        if img.endswith('.png'):
            images.append(Image.open(img))
        else:
            imgs.remove(img)
            
    images[0].save('test.gif',
               save_all=True,
               append_images=images[1:],
               duration=100,
               loop=0)

    # os.system('convert -loop 0 %s anime.gif' % ' '.join(imgs))

    # with imageio.get_writer(os.getcwd(), mode='I') as writer:
    #     for filename in imgs:
    #         if filename.endswith('.png'):
    #             image = imageio.imread(filename)
    #             writer.append_data(image)

if __name__ == "__main__":
    img2gif()