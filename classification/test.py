import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

def _test(model, epoch, testloader, device, NUM_CLS, recoder=None, writer=None, name="Test"):
    model.eval()
    all_ground_truth = []
    all_predictions = []
    with torch.no_grad():
        for data in testloader:
            text, labels = data[0].to(device), data[1].to(device)
            outputs = model(text)
            temp, predicted = torch.max(outputs.data, 1)
            all_ground_truth.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    if NUM_CLS > 2:
        acc = accuracy_score(all_ground_truth, all_predictions)
        macro_f1 = f1_score(all_ground_truth, all_predictions, average='macro')
        micro_f1 = f1_score(all_ground_truth, all_predictions, average='micro')
        weighted_f1 = f1_score(all_ground_truth, all_predictions, average='weighted')
        print(f"[{name}] Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
        if name == "Test":
            recoder.update(macro_f1, epoch, model, acc=acc, micro_f1=micro_f1, weighted_f1=weighted_f1)
        if writer:
            writer.add_scalar('Accuracy/{}'.format(name), acc, epoch)
            writer.add_scalar('Macro F1/{}'.format(name), macro_f1, epoch)
            writer.add_scalar('Micro F1/{}'.format(name), micro_f1, epoch)
            writer.add_scalar('Weighted F1/{}'.format(name), weighted_f1, epoch)
    
    elif NUM_CLS == 2:
        acc = accuracy_score(all_ground_truth, all_predictions)
        f1 = f1_score(all_ground_truth, all_predictions, average='binary')
        print(f"[{name}] Accuracy: {acc:.4f}, F1: {f1:.4f}")
        if name == "Test":
            recoder.update(f1, epoch, model, acc=acc)
        if writer:
            writer.add_scalar('Accuracy/{}'.format(name), acc, epoch)
            writer.add_scalar('F1/{}'.format(name), f1, epoch)
            
def report(all_labels, all_predictions, n_class):
    target_names = [f'Class {i}' for i in range(n_class)]
    report_str = classification_report(
        all_labels, all_predictions, zero_division=0, labels=list(range(n_class)), target_names=target_names, digits=4)
    print(report_str)
    bin_all_labels = [1 if label > 0 else label for label in all_labels]
    bin_all_predictions = [1 if label > 0 else label for label in all_predictions]
    accuracy = accuracy_score(bin_all_labels, bin_all_predictions)
    print('Normal Accuracy: %.2f %%' % (100 * accuracy), '\n')
    return

def test(model, testloader, device, NUM_CLS):
    model.eval()
    all_ground_truth = []
    all_predictions = []
    with torch.no_grad():
        for data in testloader:
            text, labels = data[0].to(device), data[1].to(device)
            outputs = model(text)
            temp, predicted = torch.max(outputs.data, 1)
            all_ground_truth.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    report(all_ground_truth,all_predictions,NUM_CLS)


def predict(model, testloader, device):
    model.eval()
    all_ground_truth = []
    all_predictions = []
    with torch.no_grad():
        for data in testloader:
            text, labels = data[0].to(device), data[1].to(device)
            outputs = model(text)
            temp, predicted = torch.max(outputs.data, 1)
            all_ground_truth.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    return all_predictions
    