import numpy as np
import torch
from PIL import Image
from model.utilities import preprocess_image, postprocess_image



def removebg(pixpath,net,device):
    # 读取图像
    img = Image.open(pixpath).convert('RGB')
    
    # 准备输入
    model_input_size = [1024, 1024]
    orig_im = np.array(img)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    # 推理
    with torch.no_grad():
        result = net(image)

    # 后处理
    result_image = postprocess_image(result[0][0], orig_im_size)

    # 保存结果
    no_bg_image = Image.new("RGBA", img.size, (0, 0, 0, 0))
    no_bg_image.paste(img, mask=Image.fromarray(result_image))

    print("处理完毕")
    return no_bg_image
