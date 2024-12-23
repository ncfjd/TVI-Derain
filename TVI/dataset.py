import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from rlp.utils import load_img, random_add_jpg_compression
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor
    
##################################################################################################
# class DatasetTrain(Dataset):
#     def __init__(self, data_dir, img_options=None):
#         super(DatasetTrain, self).__init__()
#
#         input_folder = 'rainy'
#         gt_folder = 'gt'
#         self.augment   = Augment_RGB_torch()
#         self.transforms_aug = [method for method in dir(self.augment) if callable(getattr(self.augment, method)) if not method.startswith('_')]
#
#         input_filenames = sorted(os.listdir(os.path.join(data_dir, input_folder)))
#         gt_filenames   = sorted(os.listdir(os.path.join(data_dir, gt_folder)))
#
#         #gt_filenames = [x[:-7]+'.png' for x in input_filenames]
#
#         self.input_paths = [os.path.join(data_dir, input_folder, x) for x in input_filenames if is_image_file(x)]
#         self.gt_paths    = [os.path.join(data_dir, gt_folder, x)    for x in gt_filenames    if is_image_file(x)]
#
#         self.img_options = img_options
#
#         self.img_num = len(self.input_paths)
#
#     def __len__(self):
#         return self.img_num
#
#     def __getitem__(self, index):
#         tar_index = index % self.img_num
#         input = torch.from_numpy(np.float32(load_img(self.input_paths[tar_index])))
#         gt    = torch.from_numpy(np.float32(load_img(self.gt_paths[tar_index])))
#
#         # input = torch.from_numpy(random_add_jpg_compression(input, [35,90]))
#
#         input = input.permute(2,0,1)
#         gt    = gt.permute(2,0,1)
#
#         input_name = os.path.split(self.input_paths[tar_index])[-1]
#         gt_name    = os.path.split(self.gt_paths[tar_index])[-1]
#
#         # Random Crop
#         ps = self.img_options['patch_size']
#         H = gt.shape[1]
#         W = gt.shape[2]
#         r = np.random.randint(0, H - ps) if H-ps>0 else 0
#         c = np.random.randint(0, W - ps) if H-ps>0 else 0
#
#         input = input[:, r:r + ps, c:c + ps]
#         gt    = gt[:, r:r + ps, c:c + ps]
#
#         apply_trans = self.transforms_aug[random.getrandbits(3)]
#
#         input = getattr(self.augment, apply_trans)(input)
#         gt    = getattr(self.augment, apply_trans)(gt)
#
#         return input, gt, input_name, gt_name
#########################################################################################
import pandas as pd
class DatasetTrain(Dataset):
    def __init__(self, data_dir, csv_path, img_options=None):
        super(DatasetTrain, self).__init__()

        input_folder = 'input'
        gt_folder = 'target'
        self.augment = Augment_RGB_torch()
        self.transforms_aug = [method for method in dir(self.augment) if callable(getattr(self.augment, method)) if
                               not method.startswith('_')]

        input_filenames = sorted(os.listdir(os.path.join(data_dir, input_folder)))
        gt_filenames = sorted(os.listdir(os.path.join(data_dir, gt_folder)))

        # gt_filenames = [x[:-7]+'.png' for x in input_filenames]

        self.input_paths = [os.path.join(data_dir, input_folder, x) for x in input_filenames if is_image_file(x)]
        self.gt_paths = [os.path.join(data_dir, gt_folder, x) for x in gt_filenames if is_image_file(x)]

        self.img_options = img_options

        self.img_num = len(self.input_paths)

        # 加载 CSV 文件
        self.df = pd.read_csv(csv_path, header=None, names=['line'])
        print(f"CSV file path: {csv_path}")
        self.text_dict = {}
        for _, row in self.df.iterrows():
            parts = row['line'].split('\t')
            if len(parts) == 2:
                filepath, text = parts
                filename = os.path.basename(filepath)
                self.text_dict[filename] = text.strip()

        # print("Text Dictionary Content:")
        # for key, value in self.text_dict.items():
        #     print(f"{key}: {value}")

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        tar_index = index % self.img_num
        input = torch.from_numpy(np.float32(load_img(self.input_paths[tar_index])))
        gt = torch.from_numpy(np.float32(load_img(self.gt_paths[tar_index])))

        # input = torch.from_numpy(random_add_jpg_compression(input, [35,90]))

        input = input.permute(2, 0, 1)
        gt = gt.permute(2, 0, 1)

        input_name = os.path.split(self.input_paths[tar_index])[-1]
        gt_name = os.path.split(self.gt_paths[tar_index])[-1]

        # 获取对应的文本
        text = self.text_dict.get(input_name, "")

        # Random Crop
        ps = self.img_options['patch_size']
        H = gt.shape[1]
        W = gt.shape[2]
        r = np.random.randint(0, H - ps) if H - ps > 0 else 0
        c = np.random.randint(0, W - ps) if H - ps > 0 else 0

        input = input[:, r:r + ps, c:c + ps]
        gt = gt[:, r:r + ps, c:c + ps]

        apply_trans = self.transforms_aug[random.getrandbits(3)]

        input = getattr(self.augment, apply_trans)(input)
        gt = getattr(self.augment, apply_trans)(gt)


        return input, gt, input_name, gt_name,text



##################################################################################################
# class DatasetTest(Dataset):
#     def __init__(self, inp_dir):
#         super(DatasetTest, self).__init__()
#
#         inp_files = sorted(os.listdir(inp_dir))
#         self.inp_paths = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]
#
#         self.inp_num = len(self.inp_paths)
#
#         self.save_dir ='D:/Python Code/RLP-main/RLP-main/logs/testresults/true'
#
#
#     def __len__(self):
#         return self.inp_num
#
#     def __getitem__(self, index):
#
#         inp_path = self.inp_paths[index]
#         filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
#         inp = Image.open(inp_path)
#         # 如果图片的高度和宽度不能被2整除，则裁剪图片
#         if inp.size[0] % 2 != 0 or inp.size[1] % 2 != 0:
#             new_height = inp.size[0] // 2 * 2
#             new_width = inp.size[1] // 2 * 2
#             inp = inp.crop((0, 0, new_height, new_width))
#             # 保存裁剪后的图片
#             save_path = os.path.join(self.save_dir, f"{filename}_cropped.jpg")
#             inp.save(save_path)
#
#
#         inp = TF.to_tensor(inp)
#         return inp, filename
class DatasetTest(Dataset):
    def __init__(self, root_dir,csv_path):
        super(DatasetTest, self).__init__()



        self.rainy_dir = os.path.join(root_dir, 'input')
        self.gt_dir = os.path.join(root_dir, 'target')

        rainy_files = sorted(os.listdir(self.rainy_dir))
        self.rainy_paths = [os.path.join(self.rainy_dir, x) for x in rainy_files if is_image_file(x)]

        gt_files = sorted(os.listdir(self.gt_dir))
        self.gt_paths = [os.path.join(self.gt_dir, x) for x in gt_files if is_image_file(x)]

        self.inp_num = len(self.rainy_paths)

        self.save_dir = 'D:/Python Code/RLP-main/RLP-main/logs/testresults/true'
        # 加载 CSV 文件
        self.df = pd.read_csv(csv_path, header=None, names=['line'])
        self.text_dict = {}
        for _, row in self.df.iterrows():
            parts = row['line'].split('\t', 1)  # 分割为两部分，限制分割次数为1
            if len(parts) == 2:
                filepath, text = parts
                filename = os.path.basename(filepath)
                self.text_dict[filename] = text.strip()  # 确保去除多余的空白符



    def __len__(self):
        return self.inp_num

    def __getitem__(self, index):
        rainy_path = self.rainy_paths[index]
        gt_path = self.gt_paths[index]

        rainy_filename = os.path.splitext(os.path.split(rainy_path)[-1])[0]+ '.png'
        gt_filename = os.path.splitext(os.path.split(gt_path)[-1])[0]+ '.png'

        rainy_img = Image.open(rainy_path)
        gt_img = Image.open(gt_path)
        # 获取对应的文本
        text = self.text_dict.get(rainy_filename, "")

        # unet如果图片的高度和宽度不能被2整除，则裁剪图片
        if rainy_img.size[0] % 2 != 0 or rainy_img.size[1] % 2 != 0:
            new_height = rainy_img.size[0] // 2 * 2
            new_width = rainy_img.size[1] // 2 * 2

            rainy_img = rainy_img.crop((0, 0, new_height, new_width))
            gt_img = gt_img.crop((0, 0, new_height, new_width))
            # 保存裁剪后的图片
            save_path = os.path.join(self.save_dir, f"{gt_filename}_cropped.jpg")
            #print(save_path)
            gt_img.save(save_path)
        #uformer
        # new_height = 256
        # new_width = 256
        # rainy_img = rainy_img.crop((0, 0, new_height, new_width))
        # gt_img = gt_img.crop((0, 0, new_height, new_width))
        # # 保存裁剪后的图片
        # save_path = os.path.join(self.save_dir, f"{gt_filename}_cropped.jpg")
        # # print(save_path)
        # gt_img.save(save_path)

        print(f"Filename: {rainy_filename}, Text: {text}")
        inp = TF.to_tensor(rainy_img)
        return inp, rainy_filename , text