import torch
import clip
from PIL import Image
import json
import os
import csv
import argparse
from tqdm import tqdm
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, default=0) # "IS" or "FID"
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    savename = f"score_CLIP.txt"
    print("save to", savename)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = int(args.gpu)
    torch.cuda.set_device(device)
    model, preprocess = clip.load("ViT-B/32", device=device)

    with open("../coco_id2caption_chinese_english_3w.json", 'r') as f:
        caption_dict = json.load(f)

    idset = set(caption_dict.keys())

    img_folder_path = "/dataset/fd5061f6/cogview/mnt/sfs_turbo/cogview2/coco_samples"
    dir_list = os.listdir(img_folder_path)
    
    cnt = 0
    # start = 8000*args.node

    # with open("score_CLIP.txt", 'w', encoding='utf8') as f:
    #     f.truncate()

    # for imgdir in tqdm(dir_list[start:start+8000]):
    for imgdir in tqdm(dir_list):

        if imgdir in idset:
            cnt += 1
            imglist = os.listdir(img_folder_path+"/"+imgdir)
            imglist = [name for name in imglist if re.match(r"[0-9]+\.jpg", name)]
            if len(imglist)!=60:
                continue
            scorelist = []
            text = clip.tokenize([caption_dict[imgdir][0],]).to(device)
            images = []
            for img in imglist:
                images.append(preprocess(Image.open(img_folder_path+"/"+imgdir+"/"+img)).unsqueeze(0).to(device))
            images = torch.cat(images, dim=0)
            with torch.no_grad():
                # image_features = model.encode_image(images)
                # text_features = model.encode_text(text)
                logits_per_image, logits_per_text = model(images, text)
                scorelist = list(logits_per_text.cpu().numpy()[0])
            with open(savename, 'a', encoding='utf8') as t:
                writer = csv.writer(t, delimiter ='\t')
                writer.writerow([caption_dict[imgdir][0],] + [img_folder_path+"/"+imgdir+"/"+img for img in imglist])
                writer.writerow(scorelist)


