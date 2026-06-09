import tensorflow as tf
import numpy as np

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
    def from_config(cls, config): return cls(**config)

m = tf.keras.models.load_model(
    r'models\best_forgery_detector_v2.keras',
    custom_objects={'CrossAttentionFusion': CrossAttentionFusion}
)

# Check output layer names and shapes
print("Output names:", [o.name for o in m.outputs])
print("Output shapes:", [o.shape for o in m.outputs])

# Check what the model predicts on dummy input
dummy = {k: np.zeros((1,128,128,3), dtype=np.float32) for k in ('input_rgb','input_ela','input_dct')}
pred = m.predict(dummy, verbose=0)
print("Raw prediction type:", type(pred))
if isinstance(pred, dict):
    for k, v in pred.items():
        print(f"  key='{k}' → shape={v.shape}, values={v}")
elif isinstance(pred, list):
    for i, v in enumerate(pred):
        print(f"  output[{i}] → shape={v.shape}, values={v}")
else:
    print(f"  shape={pred.shape}, values={pred}")