
import os
import json
from data_utils.unified_tokenizer import UnifiedTokenizer
import numpy as np
from tqdm import tqdm

ks = [1, 2, 4, 8, 16, 32, 60]

top_score_inks = [[], [], [], [], [], [], []]
token_nums = []
captions = []
ids = set()

if __name__ == "__main__":
    # with open('../chosen_ids.txt', 'r', encoding='utf8') as f:
    #     while True:
    #         line = f.readline().strip('\n')
    #         if line == "":
    #             break
    #         ids.add(line)

    tokenizer = UnifiedTokenizer(
            "/dataset/fd5061f6/cogview/vqvae_hard_biggerset_011.pt",
            device=7,
            img_tokenizer_num_tokens=8192
            )

    
    cnt = 0
    scorepath = "/workspace/dm/SwissArmyTransformer/coco_scores/"
    scorefiles = os.listdir(scorepath)
    scorefiles = [f for f in scorefiles if os.path.isfile(scorepath+f)]
    for filename in tqdm(scorefiles):
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
                id2 = nameline[-1].split("/")[1]
                # if id2 not in ids:
                #     continue
                caption = "".join(nameline[:-60])
                captions.append(caption)
                token_nums.append(len(tokenizer.EncodeAsIds(caption))) # divided by token nums

                for i in range(7):
                    top_score_inks[i].append(max(scoreline[:ks[i]]))

    # token_num_dict = {}
    # token_num_dict["token num"] = token_nums
    # with open("tokennum.json", 'w') as f:
    #     json.dump(token_num_dict, f)

    import math
    for i in tqdm(range(7)):
        base = 0.
        res = []
        for j in range(len(top_score_inks[i])):
            s =  math.exp((top_score_inks[i][j])/ (token_nums[j]))
            res.append(s)
        print(f"k={ks[i]}, ave=", np.average(res), "var=", np.var(res))
        
        # ave(e^{score/tokennum})

