
import os
import json
from data_utils.unified_tokenizer import UnifiedTokenizer
import numpy as np

ks = [1, 2, 4, 8, 16, 32, 60]

top_score_inks = [[], [], [], [], [], [], []]
token_nums = []
captions = []
ids = set()

if __name__ == "__main__":
    with open('../chosen_ids.txt', 'r', encoding='utf8') as f:
        while True:
            line = f.readline().strip('\n')
            if line == "":
                break
            ids.add(line)

    tokenizer = UnifiedTokenizer(
            "/dataset/fd5061f6/cogview/vqvae_hard_biggerset_011.pt",
            device=7,
            img_tokenizer_num_tokens=8192
            )

    
    cnt = 0
    with open('../score_all.txt', 'r', encoding='utf8') as f:
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
            token_nums.append(len(tokenizer.EncodeAsIds(caption)))
            # token_nums.append(len((caption)))

            for i in range(7):
                top_score_inks[i].append(max(scoreline[:ks[i]]))

    # token_num_dict = {}
    # token_num_dict["token num"] = token_nums
    # with open("tokennum.json", 'w') as f:
    #     json.dump(token_num_dict, f)

    import math
    for i in range(7):
        base = 0.
        res = []
        for j in range(len(top_score_inks[i])):
            s =  math.exp(top_score_inks[i][j]/ token_nums[j])
            res.append(s)
        print(f"k={ks[i]}, ave=", np.average(res), "var=", np.var(res))


    # # final = {}
    # # final["gt"] = gt_files
    # # for i in range(7):
    # #     final[str(ks[i])] = top1inks[i]
    # # with open("selected2.txt", 'w') as f:
    # #     json.dump(final, f)
    # # print("cnt = ", cnt)
    # import math
    # for i in range(7):
    #     base = 0.
    #     for s in top_score_inks[i]:
    #         base += s
    #     res = (math.exp(base/(len(top_score_inks[i]))))
    #     print(f"k={ks[i]}, score={res}")

