
from gaussianimage_rs import GaussianImage_RS # 替换为你实际的模型类
import torchvision.transforms as transforms
from PIL import Image
from train import *
import torch

def save_2dgs_params(model, save_path="2dgs_params.pth"):
    params = {
        "num_points": model.init_num_points,
        "H": model.H,
        "W": model.W,
        "BLOCK_W": model.BLOCK_W,
        "BLOCK_H": model.BLOCK_H,
        "tile_bounds": model.tile_bounds,
        "device": str(model.device),
        "_xyz": model._xyz.detach().cpu(),
        "_scaling": model._scaling.detach().cpu(),
        "_opacity": model._opacity.detach().cpu(),
        "_rotation": model._rotation.detach().cpu(),
        "_features_dc": model._features_dc.detach().cpu(),
        "background": model.background.cpu(),
    }
    torch.save(params, save_path)
    print(f"2DGS parameters saved to {save_path}")

# 假设 model 是 GaussianImage_RS 的实例
# save_3dgs_params(model)

argv = sys.argv[1:]
args = parse_args(argv)
# Cache the args as a text string to save them in the output dir later
args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
image_h, image_w = 0, 0
# if args.data_name == "kodak":
#     image_length, start = 24, 0
# elif args.data_name == "DIV2K_valid_LRX2":
#     image_length, start = 100, 800

image_path = args.dataset
# image_path = Path(args.dataset) /  f'{800 + 1:04}x2.png'

trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)

# 初始化模型（确保结构与训练时一致）
# model = GaussianImage_Cholesky()
checkpoint = torch.load(r"F:\3DGS_code\GaussianImage\checkpoints\DIV2K_valid_LRX2\GaussianImage_RS_50000_2000\0801x2\gaussian_model.pth.tar", map_location="cpu")
trainer.gaussian_model.load_state_dict(checkpoint)
trainer.gaussian_model.eval()
# torch.save(trainer.gaussian_model, "RSmodel_complete.pth")
# 读取测试图像
# img = Image.open("test_image.jpg").convert("RGB")
# with torch.no_grad():
#         out = trainer.gaussian_model()
# transform = transforms.ToPILImage()
# img = transform(out["render"].float().squeeze(0))
# name = "output/img4.png" 
# img.save(name)

save_2dgs_params(trainer.gaussian_model, save_path="2dgs_params.pth")

# def tensor_to_pil(tensor):
#     to_pil = transforms.ToPILImage()
#     return to_pil(tensor.cpu())

# # 假设 out_img 的 shape 为 [batch, 3, H, W]，取第一张图片
# out_img = img # 去掉 batch 维度

# # 转换为 PIL 图片
# pil_img = tensor_to_pil(out_img)
# pil_img.show()  # 显示图片
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])
# img_tensor = transform(img).unsqueeze(0)

# # 进行推理
# with torch.no_grad():
#     output = model(img_tensor)

# 你可以将 `output` 转换为 NumPy 数组进行进一步可视化
