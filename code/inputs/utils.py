import albumentations as A
import cv2

from inputs import sources, datasets
from torch.utils.data import DataLoader, random_split
from random import shuffle

 
# from https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/transforms/albu.py
def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized

# from https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/transforms/albu.py
class IsotropicResize(A.DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")

def get_num_categories(dataset_name):
    if dataset_name == "celebdf":
        return len(sources.CelebDF.categories)
    elif dataset_name in ["ff++", "ff++raw", "ff++high", "ff++low"]:
        return len(sources.FaceForensics.categories)
    elif dataset_name == "dfdc":
        return len(sources.DFDC.categories)
    elif dataset_name == "dfdc_preview":
        return len(sources.DFDC_preview.categories)
    elif dataset_name == "openforensics":
        return len(sources.OpenForensics.categories)
    else:
        raise NotImplementedError(f"Unknown dataset {dataset_name}")

def get_categories(dataset_name):
    if dataset_name == "celebdf":
        return sources.CelebDF.categories
    elif dataset_name == "ff++":
        return sources.FaceForensics.categories
    elif dataset_name == "dfdc":
        return sources.DFDC.categories
    elif dataset_name == "dfdc_preview":
        return sources.DFDC_preview.categories
    elif dataset_name == "openforensics":
        return sources.OpenForensics.categories
    else:
        raise NotImplementedError(f"Unknown dataset {dataset_name}")

def get_augmentation(augmentation_type, augmentation_args, target_size):
    if augmentation_type == "none":
        return None
    
    elif augmentation_type == "compression":
        if len(augmentation_args) == 0:
            ql = 30
            qu = 100
            p = 0.8
        if len(augmentation_args) == 1:
            ql = int(augmentation_args[0])
            qu = 100
            p = 0.8
        if len(augmentation_args) == 2:
            ql = int(augmentation_args[0])
            qu = int(augmentation_args[1])
            p = 0.8
        if len(augmentation_args) == 3:
            ql = int(augmentation_args[0])
            qu = int(augmentation_args[1])
            p = float(augmentation_args[2])            
            
        return A.ImageCompression(quality_lower=ql, quality_upper=qu, p=p)
    
    elif augmentation_type == "rotation":
        return A.Rotate(always_apply=False, p=1.0, limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False)
    
    elif augmentation_type == "simple":
        return A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=(0,3), sigma_limit=(0.5,3), p=0.1),
            A.HorizontalFlip(),
            A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
            A.ToGray(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5)
        ])

    elif augmentation_type == "selimsef":
        size=int(target_size[0]) # args.size refers to the model 1d size (e.g., 300)
        return A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.05),
            A.HorizontalFlip(),
            A.OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
            A.ToGray(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ])

    else:
        raise NotImplementedError(f"Unknown augmentation type \'{augmentation_type}\'")    
        
        
# for future use
def get_preprocessing(preprocessing_type, args):

    if preprocessing_type == "none":
        return None

    else:
        raise NotImplementedError(f"Unknown preprocessing type \'{preprocessing_type}\'")    

def get_dataset(name,
                target_size=(224, 224),
                augmentation=None,
                margin=None,
                output_type={"task": "attribute"},
                mode=None,
                qualities=None,
                categories=None,
                split=None,
                preprocessing=None,
                max_num_samples=None):
    
    if (mode is not None) and (mode not in ["train", "val", "test"]):
        raise Exception(f"Unknown mode '{mode}', only 'train', 'val', 'test' are supported")
    
    margin = margin or 1.3
    small_margin = (margin == 1.3)
    

    if name == "ff++":
        
        qualities = qualities or ["raw", "low", "high"]
        assert all([quality in ["raw", "low", "high"] for quality in qualities]), \
                f"Unknown qualities in {qualities}, must be 'raw', 'high', 'low'"
        if categories is not None:
            assert all([category in sources.FaceForensics.categories for category in categories]), \
                f"Unknown categories in {categories}, must be {sources.FaceForensics.categories}"
                    
        source = sources.FaceForensics(small_margin=small_margin, categories=categories, qualities=qualities)

        if mode is not None:
            assert mode in ["train", "val", "test"], f"Unknown mode '{mode}', only 'train', 'val', 'test' are supported"
            source = source.official_split(mode)

        data = source.discriminative_samples("attribution")

        
    elif name == "ff++raw":
        
        if qualities is not None:
            print(f"ff++raw was requested, ignoring specified qualities {qualities}")
        if categories is not None:
            assert all([category in sources.FaceForensics.categories for category in categories]), \
                f"Unknown categories in {categories}, must be {sources.FaceForensics.categories}"
 
        source = sources.FaceForensics(small_margin=small_margin, categories=categories, qualities=["raw"])

        if mode is not None:
            assert mode in ["train", "val", "test"], f"Unknown mode '{mode}', only 'train', 'val', 'test' are supported"
            source = source.official_split(mode)

        data = source.discriminative_samples("attribution")


    elif name == "ff++high":
        
        if qualities is not None:
            print(f"ff++high was requested, ignoring specified qualities {qualities}")
        if categories is not None:
            assert all([category in sources.FaceForensics.categories for category in categories]), \
                f"Unknown categories in {categories}, must be {sources.FaceForensics.categories}"

        source = sources.FaceForensics(small_margin=small_margin, categories=categories, qualities=["high"])
  
        if mode is not None:
            assert mode in ["train", "val", "test"], f"Unknown mode '{mode}', only 'train', 'val', 'test' are supported"
            source = source.official_split(mode)
        data = source.discriminative_samples("attribution")

    elif name == "ff++low":
        
        if qualities is not None:
            print(f"ff++low was requested, ignoring specified qualities {qualities}")
        if categories is not None:
            assert all([category in sources.FaceForensics.categories for category in categories]), \
                f"Unknown categories in {categories}, must be {sources.FaceForensics.categories}"

        source = sources.FaceForensics(small_margin=small_margin, categories=categories, qualities=["low"])

        if mode is not None:
            assert mode in ["train", "val", "test"], f"Unknown mode '{mode}', only 'train', 'val', 'test' are supported"
            source = source.official_split(mode)

        data = source.discriminative_samples("attribution")
        
    elif name == "celebdf":

        categories = categories or sources.CelebDF.categories

        assert all([category in sources.CelebDF.categories for category in categories]), \
            f"Celebdf categories should be {sources.CelebDF.categories}, received {categories}"
        if qualities is not None:
            print("warning: celebdf does not have different qualities, ignoring qualities argument")
        if (split is not None) or (mode is not None):
            print("warning: celebdf currently does not support splitting, returning the whole dataset")
        

        source = sources.CelebDF(small_margin, categories=categories)
        data = source.samples_for_attribution()
        
    elif name == "dfdc_preview":
        if qualities is not None:
            print("warning: dfdc_preview does not have different qualities, ignoring qualities argument")
        if split is not None:
            print("warning: dfdc_preview currently does not support arbitrary splits")
        if categories is not None:
            assert all([category in sources.DFDC_preview.categories for category in categories]), \
                f"Check your categories, must be {sources.DFDC_preview.categories}"
        modes = [mode] if mode is not None else None
        source = sources.DFDC_preview(small_margin=small_margin, modes=modes, categories=categories)
        data = source.discriminative_samples("attribution")
    
    elif name == "dfdc":
        if qualities is not None:
            print("warning: dfdc does not have different qualities, ignoring qualities argument")
        if split is not None:
            print("warning: dfdc currently does not support arbitrary splits")   
            
        if mode is not None:
            source = sources.DFDC(mode, categories=categories)
            data = source.samples_for_attribution()

        else:
            data = []
            for mode in ["train", "val", "test"]:
                source = sources.DFDC(mode, categories=categories)
                data.extend(source.samples_for_attribution())

    elif name == "openforensics":
        if qualities is not None:
            print("warning: openforensics does not have different qualities, ignoring qualities argument")
        if split is not None:
            print("warning: openforensics currently does not support arbitrary splits")
            
        if mode is not None:
            source = sources.OpenForensics(mode, small_margin, categories=categories)
            data = source.samples_for_attribution()
        else:
            data = []
            for mode in ["train", "val", "test"]:
                source = sources.OpenForensics(mode, small_margin, categories=categories)
                data.extend(source.samples_for_attribution())

    else:
        raise NotImplementedError(f"Unknown dataset {name}")

    max_num_samples = max_num_samples or len(data)
    shuffle(data)
    data = data[:max_num_samples]

    num_labels = len(source.categories)
    if output_type["task"] == "detect_fake":
        data = [(path, int(label>0)) for path, label in data]
        num_labels = 2
    elif output_type["task"] == "detect_category":
        data = [(path, int(label == output_type["category"])) for path, label in data]
        num_labels = 2
    elif output_type["task"] != "attribute":
        raise NotImplementedError("output task should be 'attribute', 'detect_fake', or 'detect_category'")
    
   
    return datasets.ImagePathDataset(data, 
                                     target_size, 
                                     augmentation=augmentation, 
                                     preprocessing=preprocessing, 
                                     margin=margin,
                                     num_labels=num_labels)


def get_dataloader(dset, batch_size, balanced=True):
    if balanced:
        return DataLoader(dset, batch_size=batch_size, num_workers=4, sampler=dset.balanced_sampler)
    else:
        return DataLoader(dset, batch_size=batch_size, num_workers=4, shuffle=True)


def split_dset(dset, weights):
    sizes = [(weight * len(dset)) // sum(weights) for weight in weights]
    sizes[-1] += len(dset) - sum(sizes)
    return random_split(dset, sizes)
