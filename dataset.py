import pathlib

import torch
import torchvision
from PIL import Image
import numpy as np



class ShogiPieceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:pathlib.Path) -> None:
        super().__init__()
        self.class_to_idx = None  # dict
        self.idx_to_class = None  # dict
        data_dict = self.load_npz(data_dir)

        data_dict = self.undersampling(data_dict, "ave")
        # return

        self.labels = []
        self.imgs = []
        self.mean = None
        self.std = None
        self._preprocess(data_dict)
        self.count = self._count(self.labels)  # dict
        self.transforms = None
        self._class_to_idx()
        self._idx_to_class()


    def load_npz(self, path_dir:pathlib.Path) -> dict:
        pieces = ["em", "fu", "gi", "hi", "ka", "ke", "ki", "ky", "ng", "nk", "ny", "ou", "ry", "to", "um"]
        d = {}
        for piece in pieces:
            path_npz = path_dir.joinpath(piece + ".npz")
            d.update(np.load(path_npz))
        return d


    def undersampling(self, data:dict, method:str="ave") -> dict:
        rng = np.random.default_rng(seed=451722)

        num = []
        for label, img in data.items():
            # print(label, img.shape[0])
            num.append(img.shape[0])
        # print(num)

        if method == "ave":
            n_max = int(sum(num) / len(num))
        elif method == "2nd":
            n_max = sorted(num)[1]  # 2nd[ia]
        else:
            n_max = sorted(num)[0]

        for label, img in data.items():
            if img.shape[0] > n_max:
                data[label] = rng.choice(data[label], n_max, replace=False)

        # for label, img in data.items():
        #     print(label, img.shape[0])
        return data


    def __getitem__(self, index:int) -> tuple[np.ndarray, str]:
        label = self.labels[index]
        img = self.imgs[index]
        img = np.array(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, self.class_to_idx[label])


    def __len__(self) -> int:
        return len(self.imgs)


    def _preprocess(self, pieces_np:dict):
        mean_sum = np.array([0, 0, 0])
        std_sum = np.array([0, 0, 0])
        
        for label, imgs in pieces_np.items():
            # print(label, imgs.shape, type(imgs))
            n_label = imgs.shape[0]
            mean, std = self._mean_std(imgs)

            if label == "em":
                mean_sum = mean_sum + mean * n_label
                std_sum = std_sum + std * n_label
                for img in imgs:
                    self.labels.append("emp")
                    self.imgs.append(img)
            
            else:
                mean_sum = mean_sum + mean * 2 * n_label
                std_sum = std_sum + std * 2 * n_label
                for img in imgs:
                    # Black piece
                    self.labels.append("b" + label)
                    self.imgs.append(img)
                    # White piece
                    self.labels.append("w" + label)
                    self.imgs.append(np.rot90(img, 2).copy())
        
        self.mean = mean_sum / len(self.imgs)
        self.std = std_sum / len(self.imgs)


    def _mean_std(self, imgs):
        '''
        imgs : (n, 64, 64, 3) (n, H, W, C) (0-255) (RGB)
        '''
        img_r = imgs[:,:,:,0]
        img_g = imgs[:,:,:,1]
        img_b = imgs[:,:,:,2]
        img_mean = [np.mean(img_r)/255, np.mean(img_g)/255, np.mean(img_b)/255]
        img_std = [np.std(img_r)/255, np.std(img_g)/255, np.std(img_b)/255]
        return np.array(img_mean), np.array(img_std)


    def _class_to_idx(self) -> None:
        cls = self.labels[:]  # deep copy
        cls = list(set(cls))
        cls.sort()
        idx = list( range( len(cls) ) )
        self.class_to_idx = dict(zip(cls, idx))


    def _idx_to_class(self) -> None:
        d = {}
        for k, v in self.class_to_idx.items():
            d[str(v)] = k
        self.idx_to_class = d


    def _count(self, labels:list[str]) -> dict:
        count = {}
        for label in labels:
            if label in count:
                count[label] += 1
            else:
                count[label] = 1
        return count




# Apply transform
class ApplyTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, ds_type:str="train", mean:list=[0.5, 0.5, 0.5], std:list=[0.5, 0.5, 0.5]):
        self.dataset = dataset
        self.mean = mean
        self.std = std

        if ds_type == "train":
            self.transform = self.train_transform()
        elif ds_type == "valid":
            self.transform = self.valid_transform()
        else:
            self.train_transform = None


    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


    def __len__(self):
        return len(self.dataset)


    def train_transform(self):
        t = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((64, 64), antialias=True),
                # torchvision.transforms.RandomResizedCrop(64),
                torchvision.transforms.RandomResizedCrop((64, 64), scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), antialias=True),
                torchvision.transforms.RandomRotation(degrees=(10)),
                torchvision.transforms.RandomErasing(),
                torchvision.transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return t

    def valid_transform(self):
        t = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((64, 64), antialias=True),
                torchvision.transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return t




def disp(img_array:np.ndarray) -> None:
    img = Image.fromarray(img_array)
    img.show()


def _sample_images(dataloader, mean:float=0.0, std:float=1.0):
    augmentation = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(degrees=(30), fill=(0.0-mean)/std),
            torchvision.transforms.RandomErasing()
        ])

    for i in range(2):
        # Load a batch of images
        imgs, _ = next(iter(dataloader))
        # imgs = augmentation(imgs)

        img = torchvision.utils.make_grid(imgs)
        img_pil = torchvision.transforms.functional.to_pil_image(img)

        # img_pil.save("sample_ds" + str(i) + ".png")
        img_pil.show()

        break


if __name__ == '__main__':
    path_current_dir = pathlib.Path(__file__).parent
    path_img_dir = path_current_dir.joinpath("images", "train_validate")

    ds = ShogiPieceDataset(path_img_dir)

    # n_images = len(ds.imgs)
    # train_size = int( n_images * 0.8 )
    # val_size = n_images - train_size
    # train_dataset, valid_dataset = torch.utils.data.random_split(ds, [train_size, val_size])

    # ds_train = ApplyTransformDataset(train_dataset, ds_type="train", mean=ds.mean, std=ds.std)
    # ds_valid = ApplyTransformDataset(valid_dataset, ds_type="valid", mean=ds.mean, std=ds.std)

    # print(len(ds_train))
    # print(len(ds_valid))


    # img, label = ds_train[0]
    # print(label)
    # print(img.shape)
    # # disp(img)



    # train_loader = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True)
    # for i, (data, target) in enumerate(train_loader):
    #     print(type(data))
    #     print(data.shape, target.shape)
    #     break


    # _sample_images(train_loader, np.average(ds.mean), np.average(ds.std))

