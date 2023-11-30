from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

pipeline.to("cuda")

image = pipeline("Dog").images[0]
image

image.save("FirstDog.png")


