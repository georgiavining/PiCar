import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torchvision.transforms as T
from picarnet import PiCarNet

#--paths----------------------------------------------------------
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))

DATA_PATH         = os.path.join(BASE_DIR, "data")
TRAIN_CSV         = os.path.join(DATA_PATH, "train.csv")
TRAIN_DIR      = os.path.join(DATA_PATH, "training_images")
TEST_DIR       = os.path.join(DATA_PATH, "test_images")
OUTPUTS_PATH    = os.path.join(BASE_DIR, "kaggle_outputs")
MODELS_DIR     = os.path.join(OUTPUTS_PATH, "models")
VISUALISATIONS_DIR = os.path.join(OUTPUTS_PATH, "visualisations")

SAVED_MODEL = 'best_model.pth'
MODEL_PATH = os.path.join(MODELS_DIR, SAVED_MODEL)


#--visualisation functions-----------------------------------------
def get_gradcam(model, image_tensor, output_idx=0):
    """
    Generate Grad-CAM for PiCarNet.
    image_tensor: (1, 3, H, W) normalised tensor
    output_idx: 0 for steering angle, 1 for speed
    """
    model.eval()
    
    gradients = []
    activations = []

    target_layer = model.backbone.features[-1]

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    
    model.zero_grad()
    output[0, output_idx].backward()

    fh.remove()
    bh.remove()

    grads = gradients[0].squeeze(0)          
    acts  = activations[0].squeeze(0)        
    
    weights = grads.mean(dim=(1, 2))        
    heatmap = (weights[:, None, None] * acts).sum(dim=0)  
    heatmap = F.relu(heatmap)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    
    return heatmap.detach().cpu().numpy()


def overlay_gradcam(original_bgr, heatmap):
    heatmap = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)

#--running visualisation------------------------------------------------------------
model = PiCarNet(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

IMG_PATH = os.path.join(TRAIN_DIR, '15.png')
img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((120, 160)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

tensor = transform(img_rgb).unsqueeze(0) 

heatmap_angle = get_gradcam(model, tensor, output_idx=0)
result_angle  = overlay_gradcam(img_bgr, heatmap_angle)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(result_angle, cv2.COLOR_BGR2RGB))
axes[1].set_title('Grad-CAM (steering angle)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(VISUALISATIONS_DIR, 'gradcam_steering.png'), dpi=150)
plt.close()


heatmap_speed = get_gradcam(model, tensor, output_idx=1)
result_speed  = overlay_gradcam(img_bgr, heatmap_speed)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')
axes[1].imshow(cv2.cvtColor(result_speed, cv2.COLOR_BGR2RGB))
axes[1].set_title('Grad-CAM (speed)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(VISUALISATIONS_DIR, 'gradcam_speed.png'), dpi=150)
plt.close()
