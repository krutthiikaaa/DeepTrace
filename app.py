"""
Workstream D — Gradio dashboard.
Calls pipeline.run() and renders the verdict badge + evidence gallery
(FFT spectrum, ELA heatmap, eye-glint crop comparison, noise map, confidence bar).
"""
import gradio as gr

from pipeline import run  # noqa: F401


def predict(image_path: str):
    raise NotImplementedError("Workstream D: call pipeline.run() and format the result for the UI")


if __name__ == "__main__":
    demo = gr.Interface(fn=predict, inputs="filepath", outputs="text")
    demo.launch()
