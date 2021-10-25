
import os
import json
from data_utils.unified_tokenizer import UnifiedTokenizer
import numpy as np
from tqdm import tqdm

ks = [1, 2, 4, 8, 16, 32, 60]
# maxnum = 60
maxnum = 1

top_score_inks = [[], [], [], [], [], [], []]
token_nums = []
zh_nums = []
captions = []
ids = set()

if __name__ == "__main__":
    ksnum = 1

    tokenizer = UnifiedTokenizer(
            "/dataset/fd5061f6/cogview/vqvae_hard_biggerset_011.pt",
            device=7,
            img_tokenizer_num_tokens=8192
            )
    
    cnt = 0
    # scorepath = "/workspace/dm/SwissArmyTransformer/coco_scores/"
    # scorepath = "/workspace/hwy/Image-cogview/baseline/caps/CogView1_1in60_coco5k_scores/"
    # scorepath = "/workspace/hwy/Image-cogview/baseline/caps/DF-GAN_punc_nodupversion_coco5k_scores/"
    # scorepath = "/workspace/hwy/Image-cogview/baseline/caps/DM-GAN_punc_nodupversion_coco5k_scores/"
    scorepath = "/workspace/hwy/Image-cogview/baseline/caps/AttnGAN_punc_nodupversion_coco5k_scores/"
    # scorepath = "/workspace/hwy/Image-cogview/baseline/caps/CogView2_1in60_coco5k_scores"
    print("score path: ", scorepath)
    scorefiles = os.listdir(scorepath)
    scorefiles = [f for f in scorefiles if os.path.isfile(scorepath+f)]
    for filename in scorefiles:
        filepath = scorepath+filename
        with open(filepath, 'r', encoding='utf8') as f:
            while True:
                nameline = f.readline()
                scoreline = f.readline()
                if nameline == "" or scoreline == "":
                    break
                cnt += 1

                nameline = nameline.split()
                scoreline = scoreline.split()
                scoreline = [float(s) for s in scoreline]
                # id2 = nameline[-1].split("/")[1]
                id2 = nameline[-1].split("/")[-1].split('.')[0]
                # if id2 not in ids:
                #     continue
                caption = "".join(nameline[:-maxnum])
                captions.append(caption)
                token_nums.append(len(tokenizer.EncodeAsIds(caption))) # divided by token nums
                zh_nums.append(len(caption))

                for i in range(ksnum):
                    top_score_inks[i].append(max(scoreline[:ks[i]]))

    # token_num_dict = {}
    # token_num_dict["token num"] = token_nums
    # with open("tokennum.json", 'w') as f:
    #     json.dump(token_num_dict, f)

    import math
    for i in range(ksnum):
        base = 0.
        res = []
        aa = []
        bb = []
        for j in range(len(top_score_inks[i])):
            # s =  math.exp((top_score_inks[i][j])/ (token_nums[j]))
            # s =  math.exp((top_score_inks[i][j])/ (zh_nums[j]))
            # s =  math.exp((top_score_inks[i][j]))
            s = (top_score_inks[i][j])/ (token_nums[j])
            # s = top_score_inks[i][j]
            res.append(s)
        print(f"k={ks[i]}, ave=", np.average(res), "var=", np.var(res))
        # print(np.sum(res)/np.sum(zh_nums))
        # ave(e^{score/tokennum})

