#pip install omegaconf
#pip install safetensors

from diffusers import UNet2DModel


def load_unet2_diffuser_model(link):
    model = UNet2DModel.from_pretrained(
        link, subfolder="unet", use_safetensors=True
    )
    return model


link = "JeffreyHuLLaMA2/DogDiffusion"

model = load_unet2_diffuser_model(link)

print(model)
