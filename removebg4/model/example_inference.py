# 该程序为示例程序，用于展示如何使用BriaRMBG模型进行图像背景移除。
# 请确保已经安装了必要的依赖库，包括torch、skimage、PIL、briarmbg、utilities和huggingface_hub。
# 如果没有安装，可以使用以下命令进行安装：

from skimage import io
import torch, os
from PIL import Image
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from huggingface_hub import hf_hub_download

def example_inference():

    im_path = f"{os.path.dirname(os.path.abspath(__file__))}/example_input.jpg"

    net = BriaRMBG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
    net.to(device)
    net.eval()    

    # prepare input
    model_input_size = [1024,1024]
    orig_im = io.imread(im_path)
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
    no_bg_image.save("example_image_no_bg.png")


if __name__ == "__main__":
    example_inference()