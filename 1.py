from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1").to("cuda")

prompt = '1个女孩,绿色头发,毛衣,看向阅图者,上半身,帽子,户外,下雪,高领毛衣'
image = pipe(prompt, guidance_scale=7.5).images[0]  
image.save("1个女孩.png")
