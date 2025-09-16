import torchvision.models
import timm
import model.timm_register_models

def easy_net_load(args):
    if args.net == 'vgg11':
        # model = timm.create_model(args.net, num_classes=args.num_classes)
        if args.is_load_imagenet:
            model = torchvision.models.vgg11(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.vgg11(weights=None)

    elif args.net == 'vgg13':
        # model = timm.create_model(args.net, num_classes=args.num_classes)
        if args.is_load_imagenet:
            model = torchvision.models.vgg13(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.vgg13(weights=None)

    elif args.net == 'vgg16':
        # model = timm.create_model(args.net, num_classes=args.num_classes)
        if args.is_load_imagenet:
            model = torchvision.models.vgg16(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.vgg16(weights=None)

    elif args.net == 'vgg19':
        # model = timm.create_model(args.net, num_classes=args.num_classes)
        if args.is_load_imagenet:
            model = torchvision.models.vgg19(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.vgg19(weights=None)

    # 后续需要加入number_class参数
    if args.net == 'squeezenet1_0':
        # model = Fourlayers()
        if args.is_load_imagenet:
            model = torchvision.models.squeezenet1_0(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.squeezenet1_0(weights=None)

    elif args.net == 'densenet121':
        # model = timm.create_model(args.net, num_classes=args.num_classes)
        if args.is_load_imagenet:
            model = torchvision.models.densenet121(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = torchvision.models.densenet121(weights=None)

    elif args.net == 'convnet_4':
        model = timm.create_model(args.net, num_classes=args.num_classes)

    return model