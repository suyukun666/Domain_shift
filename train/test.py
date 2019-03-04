import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from dataset.data_loader import myImageFloder
from torchvision import datasets
import torchnet.meter as meter 

m = meter.mAPMeter()

def test(dataset_name, epoch):
    assert dataset_name in ['VOCdevkit', 'VOCRTTS']

    model_root = os.path.join('..', 'models')
    image_root = os.path.join('..', 'dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 2
    image_size = 224
    alpha = 0

    """load data"""

    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    if dataset_name == 'VOCRTTS':
        test_list = os.path.join(image_root, 'VOCRTTS_test_labels.txt')

        dataset = myImageFloder(
            root=os.path.join(image_root, 'VOCRTTS_test'),
            label=test_list,
            transform=img_transform
        )
    else:
        test_list = os.path.join(image_root, 'VOCdevkit_test_labels.txt')
        
        dataset = myImageFloder(
            root=os.path.join(image_root, 'VOCdevkit_test'),
            label=test_list,
            transform=img_transform
        )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'VOC_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    # n_total = 0
    # n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label, requires_grad = False)

        class_output, _ = my_net(input_data=inputv_img, alpha=alpha)
        out_class_output = Variable(class_output, requires_grad = False)
        # pred = class_output.data.max(1, keepdim=True)[1]
        # n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        # n_total += batch_size

        m.add(out_class_output,classv_label)
        mAP=m.value().item()

        i += 1

    # accu = n_correct.item() * 1.0 / n_total

    print ('epoch: %d, mAP of the %s dataset: %f' % (epoch, dataset_name, mAP))
