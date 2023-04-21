import torch
import torchvision



def network(model:str="mobilenet_v3_large", num_classes:int=29):
    if model == "mobilenet_v2":
        net = torchvision.models.mobilenet_v2()
    elif model == "mobilenet_v3_small":
        net = torchvision.models.mobilenet_v3_small()
    elif model == "mobilenet_v3_large":
        net = torchvision.models.mobilenet_v3_large()
    elif model == "efficientnet_v2_s":
        net = torchvision.models.efficientnet_v2_s()
    else:
        return None
    
    if num_classes is not None:
        net.classifier[-1] = torch.nn.Linear(net.classifier[-1].in_features, out_features=num_classes)

    return net



if __name__ == '__main__':
    dummy_data = torch.randn(81, 3, 64, 64)  # 0-1

    # model = network(model="mobilenet_v3_large", num_classes=29)
    model = network(model="efficientnet_v2_s", num_classes=29)
    model.eval()
    output = model(dummy_data)

    print(model)
    print(type(model))

    print(output)
    print(output.shape)


    # for name, param in model.named_parameters():
    #     print('name  : ', name)
    #     # print('param : ', param)

