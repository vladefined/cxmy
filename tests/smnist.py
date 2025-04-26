import torch
from torch import nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import C1M2


""" Sample code of training C1M2 architecture on sMNIST / pMNIST

Achieves around 95% accuracy on validation in 20 epoch with current training settings (on sMNIST)"""


torch.manual_seed(42)

sequential_mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1).unsqueeze(-1))
])

permute = torch.randperm(28 * 28)

permuted_mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)[permute].unsqueeze(-1))
])

current_transform = sequential_mnist_transform  # sequential_mnist_transform or permuted_mnist_transform

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=current_transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=current_transform)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2048)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CXMYClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Linear(1, embedding_dim)
        self.cxmy = C1M2(embedding_dim, hidden_dim, 2, activation_fn=nn.Identity())
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.cxmy(x)[0][:, -1]
        return self.fc(x)


model = CXMYClassifier(32, 32).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

num_epochs = 20

print(f"Total examples: {len(train_loader)}")


def count_parameters(m):
    return sum(p.numel() for p in m.parameters())


print(f"Total model parameters: {count_parameters(model)}")

step = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1
        accuracy = (torch.argmax(outputs, dim=-1).squeeze(-1) == targets).float().mean()
        total_accuracy += accuracy

        if step % 10 == 0:
            print(f"Step {step} - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")

    print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    total_val_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            total_val_loss += loss.item()
            total_accuracy += (torch.argmax(outputs, dim=-1).squeeze(-1) == targets).float().mean()
    avg_val_loss = total_val_loss / len(test_loader)
    print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f} - Total Accuracy: {(total_accuracy / len(test_loader)):.4f}")
