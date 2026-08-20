import json, zipfile
from pathlib import Path
import numpy as np
import torch
from train_embedding_mlp_classifier import MLPClassifier, normalize
from cloud_20260714_seen_dino_router import prototypes, aligned_text, fused_topk
from cloud_20260714_seen_tri_router import ROOT, feats, mdl

OUT=Path('runs/cloud_20260714/seen_supervised_fivecrop_router');OUT.mkdir(parents=True,exist_ok=True)
C=ROOT/'concat_balanced_gate'; ck=torch.load(C/'mlp_h4096_balsoft/best_model.pt',map_location='cpu',weights_only=False);device=torch.device('cuda')
model=MLPClassifier(ck['arch']['in_dim'],ck['arch']['hidden_dim'],len(ck['classes']),ck['arch']['dropout']).to(device);model.load_state_dict(ck['state_dict']);model.eval()
def infer(x):
 out=[]
 with torch.inference_mode():
  for s in range(0,len(x),256):out.append(model(normalize(x[s:s+256]).to(device)).cpu())
 return torch.cat(out)
train=torch.load(C/'random2027/train.pt',map_location='cpu',weights_only=False);test=torch.load(C/'test_seen_hflip_letterbox_concat.pt',map_location='cpu',weights_only=False);five=torch.load('runs/cloud_20260714/bioclip_fivecrop_public/test_seen.pt',map_location='cpu',weights_only=False);assert test['image_ids']==five['image_ids']
orig=infer(test['features']);alt_input=torch.cat([five['features'],test['features'][:,1024:]],1);alt=infer(alt_input);blend=.5*orig+.5*alt
proto=prototypes(train['features'][:,:1024],train['class_ids'].long(),len(ck['classes']));text=aligned_text(Path('work/clip_text_features/seen_bioclip25_taxon.pt'),ck['classes']);vals,inds=fused_topk(blend,test['features'][:,:1024],proto,text,device);palt={'topk_values':vals,'topk_indices':inds,'classes':ck['classes'],'image_ids':test['image_ids']};torch.save(palt,OUT/'public_alt_topk.pt')
base=torch.load(C/'fixed_fusion_logits.pt',map_location='cpu',weights_only=False);valt=torch.load('runs/cloud_20260714/bioclip_fivecrop_priority/supervised_fused_val.pt',map_location='cpu',weights_only=False);da=torch.load(ROOT/'dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt',map_location='cpu',weights_only=False);db=torch.load(ROOT/'dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt',map_location='cpu',weights_only=False);x,b,c,_,_=feats(base,valt,da,db,base['full_class_counts'].long());truth=base['class_ids'];y=(c.eq(truth).long()-b.eq(truth).long()).numpy();reg=mdl().fit(x,y)
pb=torch.load(OUT.parent/'seen_dino_router/public_fusion.pt',map_location='cpu',weights_only=False);pa=torch.load(ROOT/'dino_metric_full_prediction/test_seen_metric_seed2027_topk.pt',map_location='cpu',weights_only=False);pd=torch.load(ROOT/'dino_metric_full_prediction/test_seen_metric_seed2028_topk.pt',map_location='cpu',weights_only=False);px,pb0,pc,_,_=feats(pb,palt,pa,pd,base['full_class_counts'].long());score=reg.predict(px)
cur=json.loads(Path('runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json').read_text(encoding='utf-8'));pos={x:i for i,x in enumerate(ck['classes'])};current=torch.tensor([pos[cur[i]] for i in pa['image_ids']]);agree=current.eq(pb0);summary={}
for pct in [5,3,2,1]:
 th=float(np.quantile(score,1-pct/100));mask=torch.from_numpy(score>=th)&agree;pred=current.clone();pred[mask]=pc[mask];out=dict(cur)
 for i,z in zip(pa['image_ids'],pred):out[i]=ck['classes'][int(z)]
 q=OUT/f'top{pct}pct';q.mkdir(exist_ok=True);f=q/'prediction.json';f.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 with zipfile.ZipFile(q/'submission.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(f,'prediction.json')
 summary[str(pct)]={'threshold':th,'selected':int(mask.sum()),'changed':sum(out[k]!=cur[k] for k in cur)}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
