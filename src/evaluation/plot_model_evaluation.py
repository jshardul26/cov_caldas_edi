"""Plot saved patient-level test predictions without retraining models."""
from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_curve, roc_curve, roc_auc_score,
                             average_precision_score)

ROOT=Path(__file__).resolve().parents[2]
FIGS=ROOT/"reports"/"figures"; FIGS.mkdir(parents=True,exist_ok=True)

def prediction_files():
    candidates=[]
    for p in (ROOT/"models"/"metadata_baselines").glob("*/test_predictions.csv"):
        candidates.append((p.parent.name,p))
    for model in ("metadata_mlp","fusion"):
        p=ROOT/"models"/model/"stage1"/"test_predictions.csv"
        if p.exists(): candidates.append((model,p))
    p=ROOT/"reports"/"baselines"/"cov_caldas_cnn_test_predictions.csv"
    if p.exists(): candidates.append(("image_resnet18",p))
    return candidates

def main():
    files=prediction_files()
    if not files: raise FileNotFoundError("No saved test predictions found. Train/evaluate models first.")
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for name,path in files:
        df=pd.read_csv(path)
        required={"label","probability_positive"}
        if not required.issubset(df.columns): raise ValueError(f"{path} must contain {sorted(required)}")
        y=df.label.astype(int).to_numpy(); prob=df.probability_positive.to_numpy()
        if len(set(y))<2: print(f"Skipping {name}: test contains one class"); continue
        fpr,tpr,_=roc_curve(y,prob); axes[0].plot(fpr,tpr,label=f"{name} (AUC {roc_auc_score(y,prob):.3f})")
        precision,recall,_=precision_recall_curve(y,prob); axes[1].plot(recall,precision,label=f"{name} (AP {average_precision_score(y,prob):.3f})")
        frac,mean=calibration_curve(y,prob,n_bins=10,strategy="quantile"); axes[2].plot(mean,frac,marker="o",label=name)
        fig_cm,ax_cm=plt.subplots(figsize=(4,4))
        ConfusionMatrixDisplay(confusion_matrix(y,prob>=.5,labels=[0,1]),display_labels=["negative","positive"]).plot(ax=ax_cm,colorbar=False)
        ax_cm.set_title(f"{name} — patient test predictions")
        fig_cm.tight_layout(); safe=re.sub(r"[^a-zA-Z0-9_-]+","_",name)
        fig_cm.savefig(FIGS/f"confusion_{safe}.png",dpi=140); plt.close(fig_cm)
    axes[0].plot([0,1],[0,1],"k--",linewidth=1); axes[0].set(xlabel="False positive rate",ylabel="True positive rate",title="ROC")
    axes[1].set(xlabel="Recall",ylabel="Precision",title="Precision–recall")
    axes[2].plot([0,1],[0,1],"k--",linewidth=1); axes[2].set(xlabel="Mean predicted probability",ylabel="Observed positive fraction",title="Calibration")
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); out=FIGS/"stage1_test_curves.png"; fig.savefig(out,dpi=160); plt.close(fig)
    print(f"Saved ROC, precision–recall, calibration plots to {out}")
    print(f"Saved one patient-level confusion matrix per model in {FIGS}")

if __name__=="__main__": main()
