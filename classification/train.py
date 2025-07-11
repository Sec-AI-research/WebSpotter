import torch
import time
from classification.test import _test

def _train_single(model, epoch, trainloader, optimizer, criterion, device, writer=None):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        text, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            if writer:
                writer.add_scalar('Loss/Train', running_loss / 50, epoch * len(trainloader) + i)
            running_loss = 0.0

def train(net, num_epochs, trainloader, testloader, optimizer, criterion, scheduler, device, NUM_CLS, rec, writer=None):

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('epoch:', epoch + 1)
        start = time.time()
        _train_single(net, epoch, trainloader, optimizer, criterion, device, writer=writer)
        _test(net, epoch, trainloader, device, NUM_CLS, recoder=rec, writer=writer, name="Train")
        elapsed = (time.time() - start)
        print("Train Time used:", elapsed)

        start = time.time()
        _test(net, epoch, testloader, device, NUM_CLS, recoder=rec, writer=writer, name="Test")
        elapsed = (time.time() - start)
        print("Test Time used:", elapsed)

        scheduler.step()
