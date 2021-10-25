import os
from PIL import Image
import PIL
import torch
from eval_utils.inception_score import inception_score
from eval_utils.fid_score import calculate_fid_given_dataset
import torchvision.transforms as transforms
import argparse



class imageFileSet(torch.utils.data.Dataset):
    def __init__(self, path, namelist, blur_r=None, transform=None, transform2=None, contrast=None):
        '''
        suffix: 对后缀的限制。筛选出符合该限制的子集
        namelist: 放入dataset的文件名（不含后缀）。若namelist is not None,必须指定后缀
        '''
        self.path = path
        self.contrast = contrast
        self.image_files = namelist
        self.blur_r = blur_r
        self.transform = transform
        self.transform2 = transform2

    def __getitem__(self, index):
        name = self.image_files[index]
        img_path = os.path.join(self.path, name)

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            self.transform(img)
        # 上一步改了con和blur顺序 现在改回来
        
        if self.blur_r is not None:
            img = img.filter(PIL.ImageFilter.GaussianBlur(radius=self.blur_r))  
        if self.contrast is not None:
            from PIL import ImageEnhance
            enh = ImageEnhance.Contrast(img)
            img = enh.enhance(self.contrast)
        # img.save("/workspace/hwy/Image-cogview/ImageEval/1.png")
            
        if self.transform2 is not None:
            img = self.transform2(img)
        return img 

    def __len__(self):
        return len(self.image_files)


def funct(namedict, blur_r=None, type="IS", key="1", image_dir=''):
    # fake_image_dir = "/dataset/fd5061f6/cogview/mnt/sfs_turbo/cogview2/" # 模型生成的图片
    # real_image_dir = "/dataset/fd5061f6/cogview/mnt/sfs_turbo/cogview2/"
    real_namelist = namedict["gt"]
    fake_namelist = namedict[key]
    
    if type == "IS":
        dataset = imageFileSet(image_dir, namelist=fake_namelist,
                                transform=transforms.Compose([transforms.Resize((256, 256)),]),
                                transform2=transforms.Compose([transforms.Resize((299, 299)),transforms.ToTensor()]), 
                                contrast=1.5, blur_r=blur_r)
        print("data len =", len(dataset.image_files))
        s_mean, s_std = inception_score(dataset, batch_size=1)
        print(f"Inception score: mean = {s_mean}, std = {s_std}")
    elif type == "FID":
        # namelist = get_intersection_img(fake_image_dir, real_image_dir, "_0.jpg", ".jpg")
        fake_dataset = imageFileSet(image_dir, namelist=fake_namelist, 
                                    transform=transforms.Compose([transforms.Resize((256, 256)),]),
                                    transform2=transforms.Compose([transforms.Resize((299, 299)),transforms.ToTensor()]), 
                                    contrast=1.5, blur_r=blur_r)
        real_dataset = imageFileSet(image_dir, namelist=real_namelist, 
                                    transform=transforms.Compose([transforms.Resize((256, 256)),]),
                                    transform2=transforms.Compose([transforms.Resize((299, 299)),transforms.ToTensor()]), 
                                    blur_r=blur_r)
        fid_value = calculate_fid_given_dataset(real_dataset, fake_dataset, batch_size=1)
        print(f"fid_value: {fid_value}")
    else:
        print("unsupported type!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="IS") # "IS" or "FID"
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--r", type=int, default=0)
    parser.add_argument("--key", type=str, default='60') # the best 1 in first {key} images

    args = parser.parse_args()
    type = args.type
    device = int(args.gpu)
    blur_r = args.r
    torch.cuda.set_device(device)
    key = args.key

    import json
    # with open("selected_caps_cogview_v1.txt", 'r') as f:
    #     namedict = json.load(f)

    # funct(namedict, blur_r=blur_r, type=type, key=key, image_dir='/dataset/fd5061f6/cogview/mnt/sfs_turbo/cogview2/')
    with open("elected_caps_cogview_v2.txt", 'r') as f:
        namedict = json.load(f)

    funct(namedict, blur_r=blur_r, type=type, key=key, image_dir='')





    

