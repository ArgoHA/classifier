from pathlib import Path
from shutil import rmtree

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from img_clf.config.resolve import CONFIG_NAME, config_dir
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from img_clf.dl.ckpt import describe_artifact
from img_clf.dl.train import prepare_model
from img_clf.dl.utils import get_latest_experiment_name


def img_preprocess(image: np.ndarray, device, img_size, mean, std) -> torch.Tensor:
    """Grad-CAM must see exactly the pixels the backends see"""
    img = cv2.resize(
        image, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA
    )  # cv2 takes (w, h); INTER_AREA to match training
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    img = (img - np.asarray(mean, np.float32)[:, None, None]) / np.asarray(
        std, np.float32
    )[:, None, None]
    return torch.from_numpy(img[None]).to(device)


def compute_gradcam(model, target_layer, img, target_class=None):
    model.eval()

    # Hook the target layer
    def save_gradient(module, grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    activations = None
    gradient = None
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)

    # Forward pass
    output = model(img)
    _, predicted = torch.max(output, 1)

    # If no specific class is targeted, use the predicted class
    if target_class is None:
        target_class = predicted.item()

    # Backward pass
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # Compute weights
    pooled_gradients = torch.mean(gradient, dim=[0, 2, 3])
    for i in range(pooled_gradients.shape[0]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.sum(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)  # ReLU removes negative values
    heatmap /= torch.max(heatmap)

    # Remove the hooks after use
    handle_forward.remove()
    handle_backward.remove()
    return heatmap


def vis_heatmap(img_pil, heatmap, output_path):
    # Resize heatmap to match the original image size
    heatmap_resized = np.array(
        Image.fromarray(heatmap.detach().numpy()).resize(img_pil.size, Image.BILINEAR)
    )

    # Create a colormap
    colormap = plt.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)

    # Convert heatmap to PIL image
    heatmap_pil = Image.fromarray((heatmap_colored[:, :, :3] * 255).astype(np.uint8))

    # Blend with original image
    blended_image = Image.blend(img_pil.convert("RGBA"), heatmap_pil.convert("RGBA"), alpha=0.4)

    # Save the resulting image
    blended_image.save(output_path)


def vis_gradcam(model, folder_to_run, output_path, target_layer, device, img_size, mean, std):
    print("Processing", folder_to_run.name)
    output_class_path = output_path / folder_to_run.name
    if output_class_path.exists():
        rmtree(output_class_path)
    output_class_path.mkdir(parents=True, exist_ok=True)
    img_paths = [
        x for x in Path(folder_to_run).glob("**/*") if x.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    for img_path in tqdm(img_paths):
        if img_path.is_file():
            img = cv2.imread(str(img_path))
            img_tensor = img_preprocess(img, device, img_size, mean, std)
            img_tensor.requires_grad_()
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            heatmap = compute_gradcam(model, target_layer, img_tensor)
            vis_heatmap(img_pil, heatmap, output_class_path / f"{img_path.stem}.png")


@hydra.main(version_base=None, config_path=config_dir(), config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    folder_to_run = Path(cfg.train.path_to_test_data)
    output_path = Path(cfg.train.visualized_path)
    ckpt_path = Path(cfg.train.path_to_save) / "model.pt"
    info = describe_artifact(ckpt_path)
    model = prepare_model(
        info["model_name"] or cfg.model_name,
        ckpt_path,
        info["num_classes"] or len(cfg.train.label_to_name),
        cfg.train.device,
    ).to("cpu")
    target_layer = model.conv_head  # last conv layer

    vis_gradcam(
        model,
        folder_to_run,
        output_path,
        target_layer,
        "cpu",
        info["img_size"] or cfg.train.img_size,
        info["mean"],
        info["std"],
    )


if __name__ == "__main__":
    main()
