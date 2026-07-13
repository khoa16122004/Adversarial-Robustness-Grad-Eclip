from util import generate_hm
from PIL import Image
from torchvision.transforms import Resize
import clip
from clip import tokenizer
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