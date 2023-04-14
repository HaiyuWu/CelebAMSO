from torchvision import transforms
import torch.utils.data as Data
import torch
from dataloader import AttributesTestDataset
import numpy as np
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser("This is testing code!")
parser.add_argument("--test_model", "-m", type=str, help="model weights")
parser.add_argument("--test_im", "-i", type=str, help="test images.")
parser.add_argument("--test_label", "-l", type=str, help="test labels")
args = parser.parse_args()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),
])

# loading dataset
print("Loading datasets...")
batch_size = 64
test_data = AttributesTestDataset(args.test_im,
                                  args.test_label,
                                  val_transform)

print("Loading model...")
test_Loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

model_net = torch.load(args.test_model, map_location="cuda:0")
model_net.eval()

print("Starting evaluation...")
total = 0
p_total = 0
n_total = 0
correct = torch.tensor([0]).cuda()
correct_p = torch.tensor([0]).cuda()
correct_n = torch.tensor([0]).cuda()
temp_result = []
with torch.no_grad():
    base = 0
    for j, (images, targets) in enumerate(tqdm(test_Loader), 0):
        images, targets = images.cuda(), targets.cuda()
        prediction = model_net(images)
        predicted = prediction.data > 0.5
        total += targets.size(0)
        n_total += len(torch.where(targets == 0)[0])
        p_total += len(torch.where(targets == 1)[0])

        base += batch_size
        for i, person in enumerate((predicted == targets), 0):
            # correct negative
            if not predicted[i] and person:
                correct_n += person
            # correct positive
            elif predicted[i] and person:
                correct_p += person
            correct += person
        temp_result = np.concatenate((temp_result, (predicted == targets).squeeze(1).cpu()))
    print(100 * correct / total)
    print(f"Acc on positive samples: {correct_p / p_total} -- {p_total} samples")
    print(f"Acc on negative samples: {correct_n / n_total} -- {n_total} samples")
