"""Local research demo for saved Cov-Caldas Stage 1 checkpoints."""
import sys, tempfile
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"/"models"))
from train_stage1_dl import DEVICE, FusionNet, MetadataMLP
sys.path.insert(0,str(ROOT/"src"/"utils"))
from image_io import set_normalization_stats, load_normalized_image

st.set_page_config(page_title="Cov-Caldas Research Demo")
st.title("Cov-Caldas COVID-19 Research Demo")
st.warning("Research use only. Not a clinical diagnostic tool; do not use for patient care.")
kind=st.selectbox("Model",["Multimodal fusion","Image-only ResNet-18","Metadata-only MLP"])
default=ROOT/("models/fusion/stage1/best.pt" if kind.startswith("Multi") else "models/checkpoints/cov_caldas_resnet18_best.pt" if kind.startswith("Image") else "models/metadata_mlp/stage1/best.pt")
path=Path(st.text_input("Checkpoint",str(default)))
age=st.number_input("Age",min_value=0.,max_value=120.,value=50.)
sex=st.selectbox("Sex",["Female","Male","Unknown"])
uploaded=st.file_uploader("Chest X-ray (fusion only)",type=["png","jpg","jpeg","tif","tiff"])

@st.cache_resource
def load_model(filename,mode):
    ck=torch.load(filename,map_location=DEVICE)
    if mode=="image":
        model=models.resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,1)
        state=ck.get("state_dict",ck)
    else:
        if ck.get("model")!=mode: raise ValueError("Selected checkpoint is for a different model")
        model=FusionNet(pretrained=False) if mode=="fusion" else MetadataMLP(); state=ck["state_dict"]
    model.load_state_dict(state); model.to(DEVICE).eval()
    norm=ck.get("image_norm_stats") or ck.get("normalization_stats")
    if mode in ("fusion","image") and norm is None:
        raise ValueError("This checkpoint lacks frozen image normalization stats. Retrain with the current CNN/fusion script.")
    if norm: set_normalization_stats(*norm)
    return model,ck

def metadata_tensor(ck):
    value={"Male":1.,"Female":0.,"Unknown":.5}[sex]; s=ck["metadata_stats"]
    return torch.tensor([[(age-s["age_mean"])/max(s["age_std"],1e-8),value]],device=DEVICE,dtype=torch.float32)

def cam_for(model,x,m,mode):
    saved={}; encoder=model.image_encoder if mode=="fusion" else model
    layer=encoder.layer4[-1]
    def hook(_,__,out):
        saved["a"]=out; out.register_hook(lambda g:saved.__setitem__("g",g))
    handle=layer.register_forward_hook(hook)
    try:
        model.zero_grad()
        logit=(model(x,m) if mode=="fusion" else model(x).squeeze(1))[0]
        logit.backward()
    finally: handle.remove()
    a,g=saved["a"].detach(),saved["g"].detach(); w=g.mean((2,3),keepdim=True)
    cam=torch.relu((w*a).sum(1,keepdim=True))
    cam=torch.nn.functional.interpolate(cam,(224,224),mode="bilinear",align_corners=False)[0,0]
    cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
    return cam.cpu().numpy(),torch.sigmoid(logit.detach()).item()

if st.button("Run model"):
    mode="fusion" if kind.startswith("Multi") else "image" if kind.startswith("Image") else "mlp"
    if not path.is_file(): st.error("Checkpoint not found. Train this model first.")
    elif mode in ("fusion","image") and uploaded is None: st.error("Upload a chest X-ray for image inference.")
    else:
        try:
            model,ck=load_model(str(path),mode); m=None if mode=="image" else metadata_tensor(ck)
            if mode=="mlp":
                with torch.no_grad(): prob=torch.sigmoid(model(m))[0].item()
            else:
                suffix=Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer()); tmp.flush(); pil=load_normalized_image(tmp.name)
                x=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(pil).unsqueeze(0).to(DEVICE)
                cam,prob=cam_for(model,x,m,mode)
                st.image(uploaded,caption="Uploaded input")
                st.image(cam,caption="Grad-CAM: positive-class evidence")
            st.metric("Predicted positive probability",f"{prob:.3f}")
            if mode=="image":
                st.info("SHAP is not applicable to this image-only model; use Grad-CAM for the image explanation.")
            else:
                try:
                    import shap
                    bg=np.asarray(ck["metadata_background"],dtype=np.float32)
                    def f(z):
                        t=torch.tensor(z,dtype=torch.float32,device=DEVICE)
                        with torch.no_grad():
                            logits=model(t) if mode=="mlp" else model(x.repeat(len(t),1,1,1),t)
                            return torch.sigmoid(logits).cpu().numpy()
                    vals=np.asarray(shap.KernelExplainer(f,bg).shap_values(m.cpu().numpy(),nsamples=30)).reshape(-1)
                    st.subheader("Metadata SHAP contributions")
                    st.write({"age":float(vals[0]),"sex":float(vals[1])})
                except ImportError: st.info("Install shap for metadata explanations.")
            st.caption("Model probability and explanations are research outputs, not medical advice.")
        except Exception as exc: st.exception(exc)
