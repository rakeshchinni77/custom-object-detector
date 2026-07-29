from pathlib import Path
from src.dataset_utils import list_image_files, read_yolo_label
from src.dataset import YOLODataset
from src.augmentations import get_train_transforms, get_valid_transforms
from src.visualization import draw_bounding_boxes, save_visualization

image_dir = Path('data/train/images')
label_dir = Path('data/train/labels')
print('images', len(list_image_files(image_dir)))
print('labels sample', read_yolo_label(label_dir / '00001.txt')[:2])
dataset = YOLODataset(image_dir=image_dir, label_dir=label_dir, transform=get_valid_transforms(320), class_names=['head','helmet','person'])
image, target = dataset[0]
print(type(image).__name__, image.shape, target['boxes'].shape, target['labels'].shape)
vis = draw_bounding_boxes(image.permute(1,2,0).numpy(), [[10,10,100,100]], ['helmet'])
output = save_visualization(vis, Path('outputs/visualizations/smoke.png'))
print(output)
