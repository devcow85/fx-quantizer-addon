import torch
from torch.utils.data import DataLoader

from torchvision import models
import torchvision.transforms as transforms

from fx_quantizer_addon import FxQuantizer, FxQuantizationEvaluator, FxNodeMapper
from fx_quantizer_addon import qfunction_set_v1, set_seed
from dvc.dataset import DVCDataset # this is custom API for loading imagenet


def load_imagenet(root, img_size = 224, batch_size = 100, num_workers = 8):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    train_set = DVCDataset(root=root, train=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

    val_set = DVCDataset(root=root, train=False, transform=val_transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_loader, val_loader


def main():
    # prepare for test
    set_seed(7) # set seed
    
    root = "/data/ImageNet2012"
    image_size = 256
    train_loader, val_loader = load_imagenet(root, img_size = image_size)  # load dataset
    
    example_input = torch.randn(1, 3, image_size, image_size)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)    # load model
    
    # fp32 model
    # sw_eval = FxQuantizationEvaluator(model, 'cuda')
    # sw_eval.evaluate(val_loader)
    
    
    # post static quantization for resnet18 model
    fx_quantizer = FxQuantizer(model, example_input)
    sw_quantization_model = fx_quantizer.ptsq(train_loader, num_batches=10)
    
    # fxq_eval = FxQuantizationEvaluator(sw_quantization_model, 'cpu')
    # fxq_eval.evaluate(val_loader)
    
    # custom quantization layer mapping for resnet18 model
    fxc_mapper = FxNodeMapper(sw_quantization_model, qfunction_set_v1)
    hw_quantization_model = fxc_mapper.map_layers()
    
    fxq_eval = FxQuantizationEvaluator(hw_quantization_model, 'cuda')
    fxq_eval.evaluate(val_loader)
    
if __name__ == "__main__":
    main()
