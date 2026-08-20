import json, zipfile
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from cloud_20260714_seen_tri_router import ROOT, feats

def mdl():
    return HistGradientBoostingRegressor(max_iter=180, learning_rate=.04, max_depth=2,
                                         min_samples_leaf=25, l2_regularization=4., random_state=2027)

OUT=Path('runs/cloud_20260714/seen_fivecrop_router'); OUT.mkdir(parents=True,exist_ok=True)
base=torch.load(ROOT/'concat_balanced_gate/fixed_fusion_logits.pt',map_location='cpu',weights_only=False)
alt=torch.load('runs/cloud_20260714/bioclip_fivecrop_priority/val_fused_topk.pt',map_location='cpu',weights_only=False)
da=torch.load(ROOT/'dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt',map_location='cpu',weights_only=False)
db=torch.load(ROOT/'dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt',map_location='cpu',weights_only=False)
x,b,c,_,_=feats(base,alt,da,db,base['full_class_counts'].long()); truth=base['class_ids']; y=(c.eq(truth).long()-b.eq(truth).long()).numpy(); model=mdl().fit(x,y)
pb=torch.load(OUT.parent/'seen_dino_router/public_fusion.pt',map_location='cpu',weights_only=False)
pa=torch.load(ROOT/'dino_metric_full_prediction/test_seen_metric_seed2027_topk.pt',map_location='cpu',weights_only=False)
pd=torch.load(ROOT/'dino_metric_full_prediction/test_seen_metric_seed2028_topk.pt',map_location='cpu',weights_only=False)
palt=torch.load('runs/cloud_20260714/bioclip_fivecrop_public/fused_topk.pt',map_location='cpu',weights_only=False)
px,pbase,pc,_,_=feats(pb,palt,pa,pd,base['full_class_counts'].long()); score=model.predict(px)
base_path=Path('runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json');cur=json.loads(base_path.read_text(encoding='utf-8'));pos={x:i for i,x in enumerate(base['classes'])};current=torch.tensor([pos[cur[i]] for i in pa['image_ids']]);agree=current.eq(pbase)
summary={};
for pct in [20,18,15,12,10,5,3,2,1]:
    threshold=float(np.quantile(score,1-pct/100));mask=torch.from_numpy(score>=threshold)&agree;pred=current.clone();pred[mask]=pc[mask];out=dict(cur)
    for i,z in zip(pa['image_ids'],pred):out[i]=base['classes'][int(z)]
    q=OUT/f'top{pct}pct';q.mkdir(exist_ok=True);f=q/'prediction.json';f.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(q/'submission.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(f,'prediction.json')
    summary[str(pct)]={'threshold':threshold,'selected':int(mask.sum()),'changed':sum(out[k]!=cur[k] for k in cur)}
torch.save({'score':torch.from_numpy(score),'base':pbase,'alternate':pc,'current':current,'image_ids':pa['image_ids']},OUT/'public_scores.pt')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
