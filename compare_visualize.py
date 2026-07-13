from util import visualize
from generate_emap import CLIPExplainRunner
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import Resize
import clip
import torch.nn.functional as F

clipmodel, preprocess = clip.load("ViT-B/16", device='cuda')

img_path = "./dog_and_car.png"
img = Image.open(img_path).convert("RGB")
caption = "a dog in a car waiting for traffic lights"

w, h = img.size
print(w,h)
resize = Resize((h,w))
                
text_processed = clip.tokenize([caption]).cuda()
# extract text featrue
text_embedding = clipmodel.encode_text(text_processed)
text_embedding = F.normalize(text_embedding, dim=-1)
print("[text embedding]:", text_embedding.shape)

explainer = CLIPExplainRunner(clipmodel=clipmodel, preprocess=preprocess, device='cuda')

hm_types = [
	'eclip',
	'eclip-wo-ksim',
	'game',
	'maskclip',
	'gradcam',
	'rollout',
    'surgery',
    'm2ib',
    'rise'
]

vis_images = []
for hm_type in hm_types:
	hm = explainer.generate_hm(hm_type, img, text_embedding, [caption], resize)
	c_ret = visualize(hm, img.copy(), resize)
	vis_images.append((hm_type, Image.fromarray(c_ret)))

font_size = max(30, min(56, w // 7))
try:
	font = ImageFont.truetype("arial.ttf", font_size)
except OSError:
	font = ImageFont.load_default()

label_h = font_size + 26
canvas_w = w * len(vis_images)
canvas_h = h + label_h
canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
draw = ImageDraw.Draw(canvas)

for idx, (name, vis_img) in enumerate(vis_images):
	x = idx * w
	canvas.paste(vis_img, (x, label_h))
	bbox = draw.textbbox((0, 0), name, font=font)
	text_w = bbox[2] - bbox[0]
	text_h = bbox[3] - bbox[1]
	text_x = x + (w - text_w) // 2
	text_y = (label_h - text_h) // 2
	draw.text((text_x, text_y), name, fill=(0, 0, 0), font=font)

canvas.save('compare_methods_row.png')

