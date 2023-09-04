import os, sys
from random import shuffle

# Example: 
#       python3 create_filelist_print.py mi-test v2 True 40k 0

exp_dir1 = sys.argv[1] if len(sys.argv) > 1 else "mi-test"
version19 = sys.argv[2] if len(sys.argv) > 1 else "v2"
if_f0_3 = sys.argv[3] == "True" if len(sys.argv) > 1 else True # 模型是否带音高指导(唱歌一定要, 语音可以不要)
sr2 = sys.argv[4] if len(sys.argv) > 1 else "40k"
spk_id5 = int(sys.argv[5]) if len(sys.argv) > 1 else 0

#print("exp_dir1=%s, version19=%s, if_f0_3=%s, sr2=%s, spk_id5=%s" % (exp_dir1, version19,if_f0_3,sr2,spk_id5) )

#version19 = "v2"
now_dir = os.getcwd()
#exp_dir1 = "mi-test"
#if_f0_3 = True
#sr2 = "40k"
#spk_id5 = 0

# 生成filelist
exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
os.makedirs(exp_dir, exist_ok=True)
gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
feature_dir = (
    "%s/3_feature256" % (exp_dir)
    if version19 == "v1"
    else "%s/3_feature768" % (exp_dir)
)
if if_f0_3:
    f0_dir = "%s/2a_f0" % (exp_dir)
    f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
    names = (
        set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
        & set([name.split(".")[0] for name in os.listdir(feature_dir)])
        & set([name.split(".")[0] for name in os.listdir(f0_dir)])
        & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
    )
else:
    names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
        [name.split(".")[0] for name in os.listdir(feature_dir)]
    )
opt = []
for name in names:
    if if_f0_3:
        opt.append(
            "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
            % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                f0_dir.replace("\\", "\\\\"),
                name,
                f0nsf_dir.replace("\\", "\\\\"),
                name,
                spk_id5,
            )
        )
    else:
        opt.append(
            "%s/%s.wav|%s/%s.npy|%s"
            % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                spk_id5,
            )
        )
fea_dim = 256 if version19 == "v1" else 768
if if_f0_3:
    for _ in range(2):
        opt.append(
            "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
            % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
        )
else:
    for _ in range(2):
        opt.append(
            "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
            % (now_dir, sr2, now_dir, fea_dim, spk_id5)
        )
shuffle(opt)
with open("%s/filelist.txt" % exp_dir, "w") as f:
    f.write("\n".join(opt))
print("write filelist done")