import glob
import os
from PIL import Image

class ImageNette:
    def __init__(self, transform, root='/raid/home/bravolu/data/imagenette2_png/val'):
        dirs = os.listdir(root)
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.data = []
        for idx, dir_ in enumerate(dirs):
            class_label = class_to_idx[dir_]
            if class_label == 9:
                fnames = glob.glob(os.path.join(root, dir_, '*.JPEG'))
                for fname in fnames:
                    print(os.path.join(root, dir_, fname))
                    #fname.replace('.png', '.JPEG')
                    img = Image.open(os.path.join(root, dir_, fname))
                    img = img.resize((224, 224))
                    png = fname.replace('.JPEG', '.png')
                    img.save(os.path.join(root, dir_, png))
                    os.remove(os.path.join(root, dir_, fname))
            else:
                fnames = glob.glob(os.path.join(root, dir_, '*.JPEG'))
            for fname in fnames:
                self.data.append(
                    [os.path.join(root, dir_, fname), class_label])
        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        return img, label


    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    #dataset = ImageNette(transform=None, root='/raid/home/bravolu/data/resize_imagenette2/train')
    dataset = ImageNette(transform=None)
