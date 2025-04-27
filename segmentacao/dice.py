import cv2
import numpy as np

#função para calcular o coeficiente Dice
def calculate_dice(mask_model, mask_ground_truth):
    mask_model = (mask_model > 0).astype(np.uint8)
    mask_ground_truth = (mask_ground_truth > 0).astype(np.uint8)

    intersection = np.sum(mask_model * mask_ground_truth)
    model_sum = np.sum(mask_model)
    ground_truth_sum = np.sum(mask_ground_truth)

    if model_sum + ground_truth_sum == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection) / (model_sum + ground_truth_sum)
    return dice

mask_model_path = 'segmentacao-rino.jpg' 
mask_ground_truth_path = 'mask_rino.png' 

mask_model = cv2.imread(mask_model_path, cv2.IMREAD_GRAYSCALE)
mask_ground_truth = cv2.imread(mask_ground_truth_path, cv2.IMREAD_GRAYSCALE)

if mask_model is None or mask_ground_truth is None:
    print("Erro ao carregar uma das imagens. Verifique o caminho.")
else:
    dice_value = calculate_dice(mask_model, mask_ground_truth)

    print(f'O coeficiente Dice entre as máscaras é: {dice_value}')