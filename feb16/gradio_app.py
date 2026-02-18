import gradio as gr
import torch
from torch import nn

model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['parameters']

def f(x):
    feature = torch.tensor([
        [x]
    ])

    X = (feature - fm) / fs

    model = nn.Linear(1,1)
    model.load_state_dict(parameters)

    prediction = model(X)
    price = prediction * ts + tm
    return price.item()

with gr.Blocks() as iface:
    x_box = gr.Number(label="Square footage")
    square_box = gr.Number(label="Price Prediction")

    x_box.change(fn=f, inputs=[x_box], outputs=[square_box])

iface.launch()