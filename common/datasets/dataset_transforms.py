from torchvision import transforms
from common.preprocessing import HistogramStretching, Equalize, ToRGBImage

def make_base_transforms(args) -> tuple:
    transform = transforms.Compose(
        [
        # RemoveImagingArtifacts() if not args.random_artifacts else RandomRemoveImagingArtifacts(random_scale=0.1),
         transforms.ToTensor(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.Resize((args.size, args.size), antialias=False),
         #transforms.RandomRotation(45, fill=0.5),
         #transforms.RandomPerspective(distortion_scale=0.4, fill=0.5),
        #  transforms.RandomCrop((args.crop_size, args.crop_size)),
         ])
    
    transform_test = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Resize((args.size, args.size), antialias=False),
        #  transforms.CenterCrop((args.crop_size, args.crop_size)),
         ])

    if args.rotation:
        transform.transforms.append(transforms.RandomRotation(args.rotation, fill=0.5))
    
    if args.perspective:
        transform.transforms.append(transforms.RandomPerspective(distortion_scale=args.perspective, fill=0.5))
    
    if args.crop_size:
        if args.random_scale_min:
            transform.transforms.append(transforms.RandomResizedCrop(args.crop_size, scale=(args.random_scale_min, 1.0)))
        else:
            transform.transforms.append(transforms.RandomCrop((args.crop_size, args.crop_size)))
        transform_test.transforms.append(transforms.CenterCrop((args.crop_size, args.crop_size)))
    
    if args.contrast:
        transform.transforms.append(transforms.RandomAutocontrast())

    if args.histogram_reference is not None:
        size = None if args.path.find("biopsies_small") == -1 else (1024, 1024)
        if args.histogram_reference.find("07_4001") != -1:
            ks = 320
        else:
            ks = 0
        transform.transforms.insert(0, Equalize(reference=args.histogram_reference, ks=ks, size=size))
        transform_test.transforms.insert(0, Equalize(reference=args.histogram_reference, ks=ks, size=size))
    
    if args.histogram_equalization:
        transform.transforms.append(HistogramStretching())
        transform_test.transforms.append(HistogramStretching())
    
    if args.rgb:
        transform.transforms.append(ToRGBImage())
        transform_test.transforms.append(ToRGBImage())
    
    if args.normalize:
        if args.rgb:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5083]
            std = [0.1061]
        transform.transforms.append(transforms.Normalize(mean, std))
        transform_test.transforms.append(transforms.Normalize(mean, std))
    
    return transform, transform_test