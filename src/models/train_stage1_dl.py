"""Stage 1 metadata MLP and multimodal image+metadata fusion training."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))
from image_io import load_normalized_image, load_and_set_norm_stats, set_normalization_stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 15
BATCH_SIZE = 16


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metadata_features(df, age_mean=None, age_std=None):
    age = pd.to_numeric(df["age"], errors="coerce")
    if age_mean is None: age_mean = float(age.mean())
    if not np.isfinite(age_mean): raise ValueError("Training ages must contain at least one valid numeric value")
    if age_std is None:
        age_std = float(age.std())
        if not np.isfinite(age_std) or age_std <= 0: age_std = 1.0
    sex = df["sex"].astype(str).str.strip().str.upper().map(
        {"M": 1.0, "MALE": 1.0, "F": 0.0, "FEMALE": 0.0}).fillna(0.5)
    x = np.column_stack([((age.fillna(age_mean) - age_mean) / age_std).to_numpy(), sex.to_numpy()]).astype("float32")
    return x, {"age_mean": age_mean, "age_std": age_std}


class Rows(Dataset):
    def __init__(self, df, x_meta, image=False, augment=False):
        self.df = df.reset_index(drop=True); self.x_meta = torch.tensor(x_meta)
        self.image = image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            *( [transforms.RandomRotation(7), transforms.ColorJitter(brightness=.1, contrast=.1)] if augment else [] ),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        y = float(row.stage1_label == "positive")
        if self.image: return self.transform(load_normalized_image(row.image_path)), self.x_meta[i], torch.tensor(y)
        return self.x_meta[i], torch.tensor(y)


class MetadataMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self, meta): return self.net(meta).squeeze(1)


class FusionNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.image_encoder = models.resnet18(weights=weights)
        n = self.image_encoder.fc.in_features; self.image_encoder.fc = nn.Identity()
        self.metadata_encoder = nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Dropout(.2))
        self.classifier = nn.Sequential(nn.Linear(n+32,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,1))
    def forward(self, image, meta):
        return self.classifier(torch.cat([self.image_encoder(image),self.metadata_encoder(meta)],dim=1)).squeeze(1)


def patient_level(df):
    cols = ["patient_id","age","sex","stage1_label","split"]
    inconsistent = df.groupby("patient_id")[cols[1:]].nunique().gt(1).any(axis=1)
    if inconsistent.any(): raise ValueError(f"Inconsistent patient metadata/labels for {int(inconsistent.sum())} patients")
    return df.groupby("patient_id",as_index=False)[cols[1:]].first().merge(
        df[["patient_id","image_path"]].drop_duplicates("patient_id"),on="patient_id",how="left")


def metrics(y, p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n":len(y),"roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            "pr_auc":float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
            "accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
            "recall_sensitivity":float(recall_score(y,pred,zero_division=0)),
            "specificity":float(tn/(tn+fp)) if tn+fp else None,"f1":float(f1_score(y,pred,zero_division=0)),
            "brier_score":float(brier_score_loss(y,p)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}


def patient_predictions(df, y, p):
    frame=pd.DataFrame({"patient_id":df["patient_id"].to_numpy(),"label":np.asarray(y).astype(int),"probability_positive":np.asarray(p)})
    if frame.groupby("patient_id")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting labels found within a patient")
    return frame.groupby("patient_id",as_index=False).agg(label=("label","first"),probability_positive=("probability_positive","mean"))


def predict(model, loader, mode):
    model.eval(); probs=[]; labels=[]
    with torch.no_grad():
        for batch in loader:
            if mode=="fusion": image,meta,y=batch; out=model(image.to(DEVICE),meta.to(DEVICE))
            else: meta,y=batch; out=model(meta.to(DEVICE))
            probs.extend(torch.sigmoid(out).cpu().numpy()); labels.extend(y.numpy())
    return np.asarray(labels),np.asarray(probs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=["mlp","fusion"],required=True)
    parser.add_argument("--epochs",type=int,default=EPOCHS)
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE)
    args=parser.parse_args(); seed_everything()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")
    mode=args.model; table=ROOT/"data/processed/cov_caldas_master_frontal.csv"
    if not table.exists(): raise FileNotFoundError(f"Missing {table}; run verification/split/frontal preprocessing first")
    images=pd.read_csv(table)
    if mode=="mlp": images=patient_level(images)
    splits={k:images[images.split==k].reset_index(drop=True) for k in ("train","val","test")}
    for k,d in splits.items():
        if d.empty or d.stage1_label.nunique()!=2: raise ValueError(f"{k} must contain both classes")
    stats_source=patient_level(splits["train"]) if mode=="fusion" else splits["train"]
    xtrain,stats=metadata_features(stats_source)
    xmap={k:metadata_features(d,stats["age_mean"],stats["age_std"])[0] for k,d in splits.items()}
    if mode=="fusion": load_and_set_norm_stats(splits["train"])
    datasets={k:Rows(d,xmap[k],mode=="fusion",k=="train") for k,d in splits.items()}
    train=splits["train"]
    if mode=="fusion":
        weights=1.0/train.patient_id.map(train.patient_id.value_counts()).to_numpy()
    else:
        weights=np.ones(len(train),dtype=np.float64)
    sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(train),replacement=True)
    loaders={k:DataLoader(v,batch_size=args.batch_size,shuffle=False if k!="train" else False,
                          sampler=sampler if k=="train" else None,num_workers=0) for k,v in datasets.items()}
    model=(MetadataMLP() if mode=="mlp" else FusionNet()).to(DEVICE)
    pos=max(1,(train.stage1_label=="negative").sum())/max(1,(train.stage1_label=="positive").sum())
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos),device=DEVICE)); opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    outdir=ROOT/"models"/("metadata_mlp" if mode=="mlp" else "fusion")/"stage1"
    outdir.mkdir(parents=True,exist_ok=True); history=[]; best=-1.; patience=0; checkpoint=outdir/"best.pt"
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in loaders["train"]:
            opt.zero_grad()
            if mode=="fusion": image,meta,y=batch; logits=model(image.to(DEVICE),meta.to(DEVICE))
            else: meta,y=batch; logits=model(meta.to(DEVICE))
            loss=loss_fn(logits,y.to(DEVICE)); loss.backward(); opt.step(); losses.append(float(loss.item()))
        vy,vp=predict(model,loaders["val"],mode)
        val_pat=patient_predictions(splits["val"],vy,vp)
        score=roc_auc_score(val_pat.label,val_pat.probability_positive)
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_roc_auc":float(score)})
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} val_auc={score:.4f}",flush=True)
        if score>best:
            best=score; patience=0
            bg_idx=np.random.RandomState(SEED).choice(len(xtrain),min(50,len(xtrain)),replace=False)
            torch.save({"model":mode,"state_dict":model.state_dict(),"metadata_stats":stats,
                        "metadata_background":xtrain[bg_idx].tolist(),
                        "image_norm_stats":__import__("image_io").get_normalization_stats() if mode=="fusion" else None,
                        "config":{"seed":SEED,"epochs":args.epochs,"batch_size":args.batch_size}},checkpoint)
        else:
            patience+=1
            if patience>=4: break
    ck=torch.load(checkpoint,map_location=DEVICE); model.load_state_dict(ck["state_dict"])
    outputs=[]
    for split,loader in loaders.items():
        y,p=predict(model,loader,mode); patient=patient_predictions(splits[split],y,p)
        row={"model":mode,"split":split,**metrics(patient.label,patient.probability_positive),"n_images":len(splits[split])}; outputs.append(row)
        patient.to_csv(outdir/f"{split}_predictions.csv",index=False)
    with open(outdir/"config.json","w") as f: json.dump({"model":mode,"metadata_features":["age","sex_M"],"sex_M_mapping":{"male":1,"female":0,"unknown":0.5},**stats,"best_val_roc_auc":best},f,indent=2)
    with open(outdir/"training_history.json","w") as f: json.dump(history,f,indent=2)
    pd.DataFrame(outputs).to_csv(outdir/"metrics.csv",index=False)
    print(f"saved checkpoint/config/history/metrics/predictions to {outdir}")


if __name__=="__main__": main()
