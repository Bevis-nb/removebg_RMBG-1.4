# from skimage import io
import numpy as np 
import torch
from PIL import Image
from model.briarmbg import BriaRMBG
from model.utilities import preprocess_image, postprocess_image
from huggingface_hub import hf_hub_download

def removebg(pixpath):
    # im_path = f"{os.path.dirname(os.path.abspath(__file__))}/example_input.jpg"
    im_path = pixpath
    img = Image.open(im_path)  
    print(img.mode)  # 应该输出 'RGB
    if img.mode !="RGB":
        rgb_image = img.convert('RGB')
    else:
        rgb_image = img
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BriaRMBG.from_pretrained("briaai/RMBG-1.4",cache_dir="./model")
    net.to(device)
    net.eval()    

    # prepare input
    model_input_size = [1024,1024]
    orig_im = np.array(rgb_image)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    # inference 
    result=net(image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
    orig_image = Image.open(im_path)
    no_bg_image.paste(orig_image, mask=pil_im)
    print("处理完毕")
    return no_bg_image
