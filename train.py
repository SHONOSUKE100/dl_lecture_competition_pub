import torch
import hydra
from omegaconf import DictConfig
from torchvision import transforms
from transformers import BertTokenizer
from torch import nn
import numpy as np
from src.utils import set_seed
from src.models.base import VQAModel
from src.datasets import VQADataset
from src.preprocs import process_text
from src.utils import VQA_criterion
import time


@hydra.main(version_base=None, config_path="configs", config_name="base")
def train(args: DictConfig):

    def train(model, dataloader, optimizer, criterion, device):
        model.train()

        total_loss = 0
        total_acc = 0
        simple_acc = 0

        start = time.time()
        for image, question, answers, mode_answer in dataloader:
            image, question, answer, mode_answer = \
                image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

            pred = model(image, question)
            loss = criterion(pred, mode_answer.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
            simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

        return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


    def eval(model, dataloader, optimizer, criterion, device):
        model.eval()

        total_loss = 0
        total_acc = 0
        simple_acc = 0

        start = time.time()
        for image, question, answers, mode_answer in dataloader:
            image, question, answer, mode_answer = \
                image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

            pred = model(image, question)
            loss = criterion(pred, mode_answer.squeeze())

            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
            simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

        return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.ToTensor()
    ])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = VQADataset(df_path="/content/drive/MyDrive/VQA/train.json", image_dir="train", transform=transform, answer=True, tokenizer=tokenizer)
    test_dataset = VQADataset(df_path="/content/drive/MyDrive/VQA/valid.json", image_dir="valid", transform=transform, answer=False, tokenizer=tokenizer)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    num_epoch = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    train()
