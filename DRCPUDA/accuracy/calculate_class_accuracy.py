import torch


def calculate_class_accuracy(y_hat, y_true):
    num_classes = y_hat.size(1)
    _, predicted = torch.max(y_hat, 1)

    correct = predicted == y_true
    total_accuracy = correct.float().mean().item()

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    for label, correct in zip(y_true, predicted == y_true):
        class_correct[label] += correct.item()
        class_total[label] += 1

    class_accuracy = [0.0] * num_classes

    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = class_correct[i] / class_total[i]

    class_accuracy.append(total_accuracy)

    return class_accuracy
