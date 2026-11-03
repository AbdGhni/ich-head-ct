"""Gradio demo: haemorrhage vs normal on a head CT slice (research prototype, not for clinical use)."""
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr

IMG = 224
MODEL = tf.keras.models.load_model("ich_effnetb0.keras")
DISCLAIMER = ("Research prototype trained on only 200 public CT slices. "
              "Not a medical device. Not for clinical use.")


def grad_cam(x):
    eff = MODEL.get_layer("efficientnetb0")
    gm = tf.keras.models.Model(eff.inputs, [eff.get_layer("top_activation").output, eff.output])
    with tf.GradientTape() as tape:
        conv, feat = gm(x)
        prob = MODEL.layers[-1](tf.reduce_mean(feat, axis=(1, 2)))[:, 0]
    g = tape.gradient(prob, conv)
    cam = tf.maximum(tf.reduce_sum(conv[0] * tf.reduce_mean(g, axis=(0, 1, 2)), -1), 0)
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()


def predict(image):
    if image is None:
        return {}, None
    rgb = np.array(image.convert("RGB").resize((IMG, IMG)), dtype=np.float32)
    x = tf.constant(rgb[None])
    p = float(MODEL.predict(x, verbose=0)[0, 0])
    overlay = None
    try:
        cam = cv2.resize(grad_cam(x), (IMG, IMG))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]
        overlay = np.uint8(0.6 * rgb + 0.4 * heat)
    except Exception:
        pass
    return {"Haemorrhage": p, "Normal": 1 - p}, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Head CT slice"),
    outputs=[gr.Label(label="Model output"), gr.Image(label="Grad-CAM (where the model looked)")],
    title="Intracranial haemorrhage on head CT (EfficientNetB0)",
    description=DISCLAIMER,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
