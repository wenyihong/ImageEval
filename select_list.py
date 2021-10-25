# import os
# import json

# id_gt = set()
# id_fake = set()
# common_id = list()

# ks = [1, 2, 4, 8, 16, 32, 60]
# top1inks = [[], [], [], [], [], [], []]
# gt_files = []
# # top_score_inks = [[], [], [], [], [], [], []]

# if __name__ == "__main__":
#     for filename in os.listdir("/dataset/fd5061f6/cogview/mnt/sfs_turbo/cogview2/groundtruth"):
#         if filename.endswith(".jpg"):
#             id_gt.add(filename.split(".")[0])
#             gt_files.append("/dataset/fd5061f6/cogview/mnt/sfs_turbo/cogview2/groundtruth/"+filename)
    
#     cnt = 0
#     scorepath = "/workspace/dm/SwissArmyTransformer/coco_scores/"
#     scorefiles = os.listdir(scorepath)
#     scorefiles = [f for f in scorefiles if os.path.isfile(scorepath+f)]
#     for filename in scorefiles:
#         filepath = scorepath+filename
#         with open(filepath, 'r', encoding='utf8') as f:
#             while True:
#                 # try:
#                 nameline = f.readline()
#                 scoreline = f.readline()
#                 if nameline == "" or scoreline == "":
#                     break
#                 cnt += 1
#                 nameline = nameline.split()[1:]
#                 scoreline = scoreline.split()
#                 scoreline = [float(s) for s in scoreline]
#                 id2 = nameline[-1].split("/")[-2]
#                 id_fake.add(id2)
#                 if id2 in id_gt:
#                     for i in range(7):
#                         # top_score_inks[i].append(max(scoreline[:ks[i]]))
#                         highest = scoreline[:ks[i]].index(max(scoreline[:ks[i]]))
#                         top1inks[i].append(nameline[-60+highest])
#                     common_id.append(id2)
#                 # except:
#                 #     break

#     final = {}
#     final["gt"] = gt_files
#     for i in range(7):
#         final[str(ks[i])] = top1inks[i]
#     with open("selected_caps.txt", 'w') as f:
#         json.dump(final, f)
#     print("cnt = ", cnt)
#     print("common ids ", len(common_id))
#     print("gt ids", len(id_gt))
#     print("generate ids", len(id_fake))
    
import os
import json

gtfiles = []
genfiles = []

for filename in os.listdir('/workspace/hwy/Image-cogview/baseline/baseline_sample_5k/DM-GAN'):
    id = filename.split('.')[0]
    genfiles.append('/workspace/hwy/Image-cogview/baseline/baseline_sample_5k/DM-GAN/'+filename)
    gtfiles.append('/dataset/fd5061f6/cogview/mnt/sfs_turbo/cogview2/groundtruth/'+id+'.jpg')
    
final = {}
final["gt"] = gtfiles
final["1"] = genfiles

with open("selected_caps_DMGAN_5k.txt", 'w') as f:
    json.dump(final, f)