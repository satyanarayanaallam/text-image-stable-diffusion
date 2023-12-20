import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
    
def image_generator(text):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    prompt = text
    image = pipe(prompt).images[0]  
    image.save("astronaut_rides_horse.png")
    return image

iface = gr.Interface(fn=image_generator, inputs="text", outputs="image")
iface.launch()