from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import io
import time
import os
import base64
import traceback
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
from scipy.fft import dctn

# ── CrossAttentionFusion (exact from training notebook) ───
class CrossAttentionFusion(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        d, h = self.embed_dim, self.num_heads
        self.attn_rgb = tf.keras.layers.MultiHeadAttention(num_heads=h, key_dim=d//h)
        self.attn_ela = tf.keras.layers.MultiHeadAttention(num_heads=h, key_dim=d//h)
        self.attn_dct = tf.keras.layers.MultiHeadAttention(num_heads=h, key_dim=d//h)
        self.norm  = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(d)
        super().build(input_shape)

    def call(self, inputs):
        rgb, ela, dct = inputs[0], inputs[1], inputs[2]
        rq = tf.expand_dims(rgb, 1)
        eq = tf.expand_dims(ela, 1)
        dq = tf.expand_dims(dct, 1)
        o1 = self.attn_rgb(rq, eq, eq)
        o2 = self.attn_ela(eq, dq, dq)
        o3 = self.attn_dct(dq, rq, rq)
        fused = tf.squeeze(o1 + o2 + o3, axis=1)
        fused = self.norm(fused + rgb)
        return self.dense(fused)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


app = Flask(__name__)

MODEL_BINARY = os.path.join('models', 'best_forgery_v5_phase2_FINAL.keras')
MODEL_TYPE   = os.path.join('models', 'best_forgery_detector_v2.keras')

print("Loading models…")
try:
    m_binary = tf.keras.models.load_model(MODEL_BINARY)
    print(f"✅ Binary model loaded | params={m_binary.count_params():,}")
except Exception as e:
    print(f"❌ Binary model failed: {e}")
    m_binary = None

try:
    m_type = tf.keras.models.load_model(
        MODEL_TYPE,
        custom_objects={'CrossAttentionFusion': CrossAttentionFusion}
    )
    print(f"✅ Type model loaded | params={m_type.count_params():,}")
except Exception as e:
    print(f"❌ Type model failed: {e}")
    m_type = None


# ── Warmup ────────────────────────────────────────────────
def warmup(model, img_size):
    dummy = {k: np.zeros((1, img_size, img_size, 3), dtype=np.float32)
             for k in ('input_rgb', 'input_ela', 'input_dct')}
    model.predict(dummy, verbose=0)

if m_binary: warmup(m_binary, 224)
if m_type:   warmup(m_type,   128)
print("✅ Warmup done")


# ── Preprocessing ─────────────────────────────────────────
def preprocess(img_bytes, img_size):
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image")

    # RGB
    rgb = cv2.resize(bgr, (img_size, img_size))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ELA — must save/reload from PIL exactly like notebook
    pil   = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pil_r = pil.resize((img_size, img_size))
    buf   = io.BytesIO()
    pil_r.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    comp    = Image.open(buf).convert('RGB')
    diff    = ImageChops.difference(pil_r, comp)
    mx      = max(e[1] for e in diff.getextrema()) or 1
    ela_pil = ImageEnhance.Brightness(diff).enhance(255.0 / mx)
    ela_arr = np.array(ela_pil)
    ela     = ela_arr.astype(np.float32) / 255.0

    # DCT
    dct_bgr = cv2.resize(bgr, (img_size, img_size))
    dct_ycr = cv2.cvtColor(dct_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    nb      = img_size // 8
    dct_out = np.zeros_like(dct_ycr)
    for ch in range(3):
        blk    = dct_ycr[:, :, ch].reshape(nb, 8, nb, 8).transpose(0, 2, 1, 3)
        d      = dctn(blk, axes=(-2, -1), norm='ortho')
        d      = np.log1p(np.abs(d))
        ch_map = d.transpose(0, 2, 1, 3).reshape(img_size, img_size)
        mx_v   = ch_map.max()
        dct_out[:, :, ch] = ch_map / mx_v if mx_v > 0 else ch_map

    sample = {
        'input_rgb': rgb[np.newaxis],
        'input_ela': ela[np.newaxis],
        'input_dct': dct_out.astype(np.float32)[np.newaxis],
    }
    return bgr, sample, ela_arr


# ── Binary prediction — mirrors notebook exactly ──────────
# notebook: pred_bin = model_binary.predict(sample_binary, verbose=0)[0]
#           label_bin = np.argmax(pred_bin)   → 0=Authentic, 1=Forged
def predict_binary(model, sample):
    raw = model.predict(sample, verbose=0)
    # m_binary is single-output ndarray, shape (1, 2)
    if isinstance(raw, dict):
        probs = next(iter(raw.values()))[0]
    elif isinstance(raw, list):
        probs = raw[0][0]
    else:
        probs = raw[0]          # shape (2,)
    return int(np.argmax(probs)), float(np.max(probs))


# ── Forgery type — mirrors notebook exactly ───────────────
# notebook: type_probs = pred_type['forgery_type'][0]
#           type_idx = np.argmax(type_probs)
#           type_names = {0:'Authentic', 1:'Copy-Move', 2:'Splicing'}
#
# KEY FIX: if argmax is 0 (Authentic), the type model is uncertain —
# pick the winner between index 1 and 2 (Copy-Move vs Splicing) instead.
def predict_type(sample_128):
    pred = m_type.predict(sample_128, verbose=0)

    # Always use 'forgery_type' head — confirmed dict output
    type_probs = pred['forgery_type'][0]   # shape (3,): [Auth, CopyMove, Splicing]

    type_idx  = int(np.argmax(type_probs))
    type_conf = float(np.max(type_probs))

    # If model picks index 0 (Authentic) but binary already confirmed forged,
    # fall back to whichever of Copy-Move / Splicing has higher probability
    if type_idx == 0:
        cm_prob  = float(type_probs[1])
        sp_prob  = float(type_probs[2])
        if sp_prob >= cm_prob:
            return 'Splicing',  sp_prob
        else:
            return 'Copy-Move', cm_prob

    type_names = {1: 'Copy-Move', 2: 'Splicing'}
    return type_names.get(type_idx, 'Unknown'), type_conf


# ── Heatmap overlay ───────────────────────────────────────
def make_heatmap_b64(bgr, ela_arr):
    h, w   = bgr.shape[:2]
    gray   = cv2.cvtColor(ela_arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (21, 21), 0)
    norm   = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    thresh = np.percentile(norm, 55)
    mask   = np.clip((norm - thresh) / (1.0 - thresh + 1e-8), 0, 1)
    mask_up = cv2.resize(mask, (w, h))

    orig   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    red    = np.zeros_like(orig); red[..., 0] = 255.0
    alpha  = (mask_up[..., np.newaxis] ** 0.75) * 0.68
    result = np.clip(orig * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)

    _, buf = cv2.imencode('.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 90])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


def encode_b64(bgr):
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify(binary=m_binary is not None, type=m_type is not None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify(error='No image uploaded'), 400
        if m_binary is None:
            return jsonify(error='Binary model not loaded'), 500

        raw_bytes = request.files['image'].read()
        if not raw_bytes:
            return jsonify(error='Empty file'), 400

        t0 = time.time()

        try:
            bgr, sample_224, ela_arr = preprocess(raw_bytes, img_size=224)
        except Exception as e:
            return jsonify(error=f'Preprocessing failed: {e}'), 400

        original_b64 = encode_b64(bgr)

        # Step 1 — Real vs Fake (m_binary, 224x224)
        label, conf = predict_binary(m_binary, sample_224)

        if label == 0:
            return jsonify(
                result='Authentic',
                confidence=round(conf * 100, 1),
                forgery_type='Real',
                inference_ms=round((time.time() - t0) * 1000),
                heatmap=None,
                original=original_b64,
            )

        # Step 2 — Copy-Move vs Splicing (m_type, 128x128)
        ftype     = 'Unknown'
        type_conf = 0.0
        if m_type is not None:
            try:
                _, sample_128, _ = preprocess(raw_bytes, img_size=128)
                ftype, type_conf = predict_type(sample_128)
            except Exception as e:
                print(f"⚠  Type prediction failed: {e}")
                traceback.print_exc()

        heatmap = make_heatmap_b64(bgr, ela_arr)
        ms      = round((time.time() - t0) * 1000)

        return jsonify(
            result='Forged',
            confidence=round(conf * 100, 1),
            forgery_type=ftype,
            type_confidence=round(type_conf * 100, 1),
            inference_ms=ms,
            heatmap=heatmap,
            original=original_b64,
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f'Server error: {e}'), 500


if __name__ == '__main__':
    app.run(debug=False, port=5000)