import json
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from cloud_20260714_seen_meta_router import ROOT, features, fold, best_threshold, guard_mask

OUT=Path('runs/cloud_20260714/seen_router_sweep');OUT.mkdir(parents=True,exist_ok=True)
v=torch.load(ROOT/'concat_balanced_gate/fixed_fusion_logits.pt',map_location='cpu',weights_only=False)
a=torch.load(ROOT/'dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt',map_location='cpu',weights_only=False)
b=torch.load(ROOT/'dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt',map_location='cpu',weights_only=False)
t=torch.load(ROOT/'triview_concat_gate/paired_random2027/fixed_fusion_taxon_logits.pt',map_location='cpu',weights_only=False)
x,base,alt=features(v['logits'],a,b,v['full_class_counts'].long(),tri=t); truth=v['class_ids'].long();y=(alt.eq(truth).long()-base.eq(truth).long()).numpy();g=np.array([fold(s) for s in v['labels']]);guard=guard_mask(v['logits'],a,b,v['full_class_counts'].long(),alt).numpy()
rows=[]
for depth in [2,3,4]:
 for leaf in [25,40,70]:
  for l2 in [4.,8.,20.]:
   o=np.zeros(len(y));ts=[];nets=[]
   for f in range(4):
    tr,te=g!=f,g==f;m=HistGradientBoostingRegressor(max_iter=180,learning_rate=.04,max_depth=depth,min_samples_leaf=leaf,l2_regularization=l2,random_state=2027).fit(x[tr],y[tr]);tt=best_threshold(m.predict(x[tr]),y[tr]);s=m.predict(x[te]);z=s>=tt;o[te]=s;ts.append(tt);nets.append(int(y[te][z].sum()))
   z=o>=np.array(ts)[g];zs=z&guard
   rows.append({'depth':depth,'leaf':leaf,'l2':l2,'changed':int(z.sum()),'net':int(y[z].sum()),'worst_fold':min(nets),'fold_nets':nets,'strict_changed':int(zs.sum()),'strict_net':int(y[zs].sum())})
rows.sort(key=lambda r:(r['worst_fold'],r['strict_net'],r['net']),reverse=True)
(OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows[:10],indent=2))
