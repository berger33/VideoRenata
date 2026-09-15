import cv2, numpy as np, mediapipe as mp, os
from scipy.spatial import Delaunay
from scipy.ndimage import uniform_filter1d

fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)

def get_lms(img):
    r = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks: return None
    h,w = img.shape[:2]
    return np.array([[p.x*w, p.y*h] for p in r.multi_face_landmarks[0].landmark[:468]], np.float32)

def load_src(path):
    im = cv2.imread(path); lms = get_lms(im); assert lms is not None, path
    return im, lms

src_n, lms_n = load_src('/tmp/swap/src_neutral.png')
src_s, lms_s = load_src('/tmp/swap/src_smile.png')
tri = Delaunay(lms_n).simplices

OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
BROWS = [70,63,105,66,107,336,296,334,293,300]
UP_LIP, LO_LIP, L_M, R_M = 13, 14, 61, 291
def mouth_open(l): return np.linalg.norm(l[UP_LIP]-l[LO_LIP])/(np.linalg.norm(l[L_M]-l[R_M])+1e-6)

def warp_face(src, src_lms, tgt_lms, shape):
    warped = np.zeros(shape, np.float32); wmask = np.zeros(shape[:2], np.float32)
    for t in tri:
        sp = src_lms[t]; tp = tgt_lms[t]
        r1 = cv2.boundingRect(sp); r2 = cv2.boundingRect(tp)
        if min(r1[2],r1[3],r2[2],r2[3]) < 1: continue
        if r2[0]<0 or r2[1]<0 or r2[0]+r2[2]>shape[1] or r2[1]+r2[3]>shape[0]: continue
        sp2=(sp-r1[:2]).astype(np.float32); tp2=(tp-r2[:2]).astype(np.float32)
        try: M = cv2.getAffineTransform(sp2[:3], tp2[:3])
        except cv2.error: continue
        patch = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        if patch.size==0: continue
        wp = cv2.warpAffine(patch, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = np.zeros((r2[3], r2[2]), np.float32)
        cv2.fillConvexPoly(m, np.int32(np.round(tp2)), 1.0)
        ys, xs = slice(r2[1], r2[1]+r2[3]), slice(r2[0], r2[0]+r2[2])
        keep = m > wmask[ys, xs]
        warped[ys, xs][keep] = wp[keep]
        wmask[ys, xs] = np.maximum(wmask[ys, xs], m)
    return warped, wmask

def swap_on_crop(crop, tgt_lms, fade=1.0):
    h,w = crop.shape[:2]
    mo = mouth_open(tgt_lms)
    a = np.clip((mo-0.02)/0.10, 0, 1)
    w_n, m_n = warp_face(src_n, lms_n, tgt_lms, crop.shape)
    w_s, m_s = warp_face(src_s, lms_s, tgt_lms, crop.shape)
    warped = w_n*(1-a) + w_s*a
    wmask = np.maximum(m_n, m_s)
    hull = np.int32(tgt_lms[OVAL])
    mask = np.zeros((h,w), np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    mask = cv2.erode(mask, np.ones((9,9),np.uint8))
    mask[wmask<0.5] = 0
    # clip forehead: keep only a modest band above the brows to avoid hairline seams
    brow_y = float(tgt_lms[BROWS][:,1].min())
    chin_y = float(tgt_lms[152][1])
    face_h = max(chin_y - brow_y, 1.0)
    top_cut = int(max(0, brow_y - 0.22*face_h))
    mask[:top_cut] = 0
    # soften the new top edge so the clone blends into forehead skin, not hair
    if mask.sum() < 255*80: return crop
    warped_u8 = np.clip(warped,0,255).astype(np.uint8)
    warped_u8[wmask<0.5] = crop[wmask<0.5]
    sel = mask>0
    lab_w = cv2.cvtColor(warped_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_t = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    mu_w, sd_w = lab_w[sel].mean(0), lab_w[sel].std(0)+1e-6
    mu_t, sd_t = lab_t[sel].mean(0), lab_t[sel].std(0)+1e-6
    sd_mix = sd_w*0.5+sd_t*0.5
    lab_c = (lab_w-mu_w)/sd_w*sd_mix+mu_t
    corr = cv2.cvtColor(np.clip(lab_c,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    corr = cv2.GaussianBlur(corr,(3,3),0)
    corr[~sel] = crop[~sel]
    ys, xs = np.where(mask>0)
    center = (int(xs.mean()), int(ys.mean()))
    try:
        out = cv2.seamlessClone(corr, crop, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        m = cv2.GaussianBlur(mask.astype(np.float32)/255,(15,15),0)
        out = np.clip(crop*(1-m[...,None])+corr*m[...,None],0,255).astype(np.uint8)
    if fade < 1.0:
        out = cv2.addWeighted(crop, 1-fade, out, fade, 0)
    return out

cap = cv2.VideoCapture('/tmp/clip_10s.mp4')
frames=[]; boxes=[]
while True:
    ok, fr = cap.read()
    if not ok: break
    frames.append(fr)
    h,w=fr.shape[:2]
    r = fd.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    if r.detections:
        d=max(r.detections,key=lambda d:d.location_data.relative_bounding_box.width)
        bb=d.location_data.relative_bounding_box
        boxes.append(((bb.xmin+bb.width/2)*w,(bb.ymin+bb.height/2)*h,max(bb.width*w,bb.height*h)))
    else: boxes.append(None)
N=len(frames)
det = np.array([b is not None for b in boxes])
print('frames',N,'det',det.sum())

idx=np.where(det)[0]
arr=np.array([boxes[i] for i in idx],np.float32)
full=np.zeros((N,3),np.float32)
for k in range(3): full[:,k]=np.interp(np.arange(N),idx,arr[:,k])
full=uniform_filter1d(full,size=9,axis=0)

valid = np.zeros(N,bool)
for i in range(N):
    valid[i] = np.abs(idx - i).min() <= 6

CROP=512
all_lms=np.zeros((N,468,2),np.float32); has=np.zeros(N,bool); crop_geo=[]
for i in range(N):
    cx,cy,s=full[i]; S=min(s*2.4, frames[i].shape[1], frames[i].shape[0])
    h,w=frames[i].shape[:2]
    x0=np.clip(cx-S/2,0,w-S); y0=np.clip(cy-S/2,0,h-S)
    crop=frames[i][int(y0):int(y0+S),int(x0):int(x0+S)]
    crop=cv2.resize(crop,(CROP,CROP),interpolation=cv2.INTER_CUBIC)
    crop_geo.append((x0,y0,S))
    if valid[i]:
        l=get_lms(crop)
        if l is not None: all_lms[i]=l; has[i]=True
hidx=np.where(has)[0]
print('mesh', has.sum(),'/',N)
for p in range(468):
    for k in range(2): all_lms[:,p,k]=np.interp(np.arange(N),hidx,all_lms[hidx,p,k])
sm=uniform_filter1d(all_lms,size=5,axis=0)
sm2=uniform_filter1d(all_lms,size=3,axis=0)
LIPS=[0,13,14,17,37,39,40,61,78,80,81,82,84,87,88,91,95,146,178,181,185,191,267,269,270,291,308,310,311,312,314,317,318,321,324,375,402,405,409,415]
EYES=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
light=list(set(LIPS+EYES))
final=sm.copy(); final[:,light]=sm2[:,light]

mvalid = np.zeros(N,bool)
for i in range(N):
    mvalid[i] = has[i] or (np.abs(hidx-i).min() <= 4)
fade = np.zeros(N,np.float32)
for i in range(N):
    if mvalid[i]:
        run=5
        for d in range(1,5):
            if i-d>=0 and not mvalid[i-d]: run=min(run,d); break
        for d in range(1,5):
            if i+d<N and not mvalid[i+d]: run=min(run,d); break
        fade[i]=min(1.0, run/4.0)

os.makedirs('/tmp/swap/outF',exist_ok=True)
for i in range(N):
    out=frames[i].copy()
    if mvalid[i] and fade[i]>0.05:
        x0,y0,S=crop_geo[i]
        crop=frames[i][int(y0):int(y0+S),int(x0):int(x0+S)]
        ch,cw=crop.shape[:2]
        crop512=cv2.resize(crop,(CROP,CROP),interpolation=cv2.INTER_CUBIC)
        res=swap_on_crop(crop512,final[i],fade=float(fade[i]))
        res_back=cv2.resize(res,(cw,ch),interpolation=cv2.INTER_AREA)
        out[int(y0):int(y0+ch),int(x0):int(x0+cw)]=res_back
    cv2.imwrite(f'/tmp/swap/outF/f_{i:04d}.png',out)
print('done. swapped:', int((mvalid&(fade>0.05)).sum()))
