import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from HSJA import *
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

trans_resize = transforms.Resize((224,224))
trans_to_tensor = transforms.ToTensor()

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
de_mean = -mean / std
de_std = 1/std
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_to_input(image):

    normalize = transforms.Normalize(mean=mean, std=std)
    input = trans_to_tensor(image)

    input = normalize(input)
    input = input.unsqueeze(0)
    return input

def output_to_image(output):

    de_normalize = transforms.Normalize(mean=de_mean, std=de_std)
    output = output.cpu().clone()
    image = output.squeeze(0)
    image = de_normalize(image)

    image = image.permute(1,2,0)
    image = image.numpy()  # squeeze和unsqueeze不是一个函数
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image

def test(net, input):
    input = input.to(device)
    with torch.no_grad():
        prediction = net(input).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")
    return category_name, round(score*100, 1)

# 对比展现原始图片和对抗样本图片
def show_images_diff(original_img, original_label, ori_score,adversarial_img, adversarial_label,adversarial_score):
    plt.figure()

    # 归一化
    if original_img.any() > 1.0:
        original_img = original_img / 255.0
    if adversarial_img.any() > 1.0:
        adversarial_img = adversarial_img / 255.0

    plt.subplot(131)
    plt.title('Ori:{},{}%'.format(original_label,ori_score))
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adv:{},{}%'.format(adversarial_label,adversarial_score))
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('perturbation')
    difference = adversarial_img - original_img
    # (-1,1)  -> (0,1)
    difference = difference / abs(difference).max() / 2.0 + 0.5
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    weights = ResNet18_Weights.DEFAULT
    ResNet18 = resnet18(weights=weights).to(device)  # 权重选择
    ResNet18.eval()

    Test_set = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True)
    random_indices = torch.randperm(len(Test_set))[:100]

    accuracy = 0
    s = 0

    for i in random_indices:
        s = s + 1
        print('Attacking example {}, the {}th attack'.format(i,s))
        ori_image, ori_label = Test_set[i]
        ori_image = trans_resize(ori_image)
        input = image_to_input(ori_image)

        attack_input = input.numpy().squeeze(0)

        pert_array = hsja(ResNet18, attack_input, clip_min=-3, clip_max=3, num_iterations=20)
        pert_tensor = torch.tensor(pert_array,dtype=torch.float32).unsqueeze(0)

        ori_pred, ori_score = test(ResNet18, input)
        pert_pred, pert_score = test(ResNet18, pert_tensor)
        pert_image = output_to_image(pert_tensor)

        show_images_diff(np.array(ori_image),ori_pred,ori_score,np.array(pert_image),pert_pred,pert_score)
        right = (pert_pred != ori_pred)
        if right == 1:
          print('successful attack! :)')
        else:
          print('unsuccessful attack :(')
        print('\n')
        accuracy = accuracy + right

    print('the total accuracy is {}% in first 100 pictures'.format(accuracy))
