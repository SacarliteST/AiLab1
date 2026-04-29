import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
import matplotlib.pyplot as plt

def rand_bbox(size: tuple, lam: float) -> tuple:
    """
    Вычисляет координаты прямоугольной области для применения аугментации CutMix.

    Args:
        size (tuple): Размеры тензора батча изображений в формате (Batch, Channels, Height, Width).
        lam (float): Коэффициент смешивания, определяющий долю площади вырезаемого фрагмента.

    Returns:
        tuple: Координаты ограничивающего прямоугольника (bounding box) в формате (bbx1, bby1, bbx2, bby2).
    """
    width = size[2]
    height = size[3]

    # Вычисление относительных размеров прямоугольника на основе коэффициента lam
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    # Генерация случайного центра прямоугольника
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    # Вычисление абсолютных координат с ограничением по краям изображения
    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    Вычисляет взвешенное значение функции потерь для аугментированных данных CutMix.

    Args:
        criterion (Callable): Базовая функция потерь (например, nn.CrossEntropyLoss).
        pred (torch.Tensor): Тензор предсказаний модели.
        y_a (torch.Tensor): Истинные метки классов исходных изображений.
        y_b (torch.Tensor): Истинные метки классов вклеенных изображений.
        lam (float): Скорректированный коэффициент смешивания площадей.

    Returns:
        torch.Tensor: Итоговое значение потерь.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model(device, model, optimizer, train_loader, train_data, val_loader, test_loader,
                criterion, num_epochs=25, use_cutmix=True):
    """
    Выполняет цикл обучения и валидации нейронной сети.

    Args:
        device (torch.device): Устройство для вычислений (CPU или GPU).
        model (torch.nn.Module): Обучаемая модель нейронной сети.
        optimizer (torch.optim.Optimizer): Алгоритм оптимизации.
        train_loader (DataLoader): Итератор обучающей выборки.
        train_data (Dataset): Обучающий набор данных.
        val_loader (DataLoader): Итератор валидационной выборки.
        criterion (Callable): Функция потерь.
        num_epochs (int, optional): Количество эпох обучения. По умолчанию 5.
        use_cutmix (bool, optional): Флаг применения аугментации CutMix. По умолчанию True.

    Returns:
        dict: История метрик (значения функции потерь на обучении и F1-меры на валидации).
    """
    history = {
        'train_loss': [],
        'val_f1': [],
        'test_epochs': [],
        'test_f1': []
    }

    for epoch in range(num_epochs):
        print(f'\nЭпоха {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # ФАЗА ОБУЧЕНИЯ
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_cutmix and np.random.rand() > 0.5:
                lam = np.random.beta(a=1.0, b=1.0)
                rand_index = torch.randperm(n=inputs.size()[0]).to(device)

                target_a = labels
                target_b = labels[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(size=inputs.size(), lam=lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                outputs = model(inputs)
                loss = cutmix_criterion(criterion=criterion, pred=outputs, y_a=target_a, y_b=target_b, lam=lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_data)
        history['train_loss'].append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')

        # ФАЗА ВАЛИДАЦИИ (на каждой эпохе)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(input=outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='weighted', zero_division=0)
        history['val_f1'].append(val_f1)
        print(f'Validation F1: {val_f1:.4f}')

        # Тестирование

        if (epoch + 1) % 5 == 0:
            all_test_preds, all_test_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(input=outputs, dim=1)
                    all_test_preds.extend(preds.cpu().numpy())
                    all_test_labels.extend(labels.cpu().numpy())

            test_f1 = f1_score(y_true=all_test_labels, y_pred=all_test_preds, average='weighted', zero_division=0)
            history['test_epochs'].append(epoch + 1)
            history['test_f1'].append(test_f1)
            print(f'TEST F1 (Эпоха {epoch + 1}): {test_f1:.4f}')

    return history


def create_model(pretrained=True):
    """
    Инициализирует архитектуру EfficientNet-B0 с адаптацией классификатора под целевую задачу.

    Args:
        pretrained (bool, optional): Флаг использования весов, предобученных на ImageNet.
                                     Если False, веса инициализируются случайным образом. По умолчанию True.

    Returns:
        torch.nn.Module: Готовая к обучению модель.
    """
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        model = models.efficientnet_b0(weights=None)

    # Адаптация выходного полносвязного слоя под 120 классов (Stanford Dogs)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=120)

    return model


def plot_learning_curves(results: dict, total_epochs: int) -> None:
    """
    Визуализирует динамику функции потерь и F1-score
    """
    epochs = range(1, total_epochs + 1)

    for exp_name, metrics in results.items():
        plt.figure(figsize=(14, 5))

        # График №1: Динамика функции потерь (Train Loss)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, metrics['train_loss'], linewidth=2.5, color='#1f77b4', label='Train Loss')

        plt.title(label=f'Сходимость функции потерь\n({exp_name})', fontsize=14)
        plt.xlabel(xlabel='Эпоха обучения', fontsize=12)
        plt.ylabel(ylabel='Значение потерь', fontsize=12)
        plt.xticks(ticks=np.arange(0, total_epochs + 1, step=5))
        plt.grid(visible=True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', fontsize=10)

        # График №2: Динамика F1-меры (Validation + Test)
        plt.subplot(1, 2, 2)

        plt.plot(epochs, metrics['val_f1'], alpha=0.5, linestyle='--', color='#ff7f0e', linewidth=2,
                 label="Validation F1")

        plt.plot(metrics['test_epochs'], metrics['test_f1'], color='#ff7f0e',
                 marker='D', markersize=8, linewidth=3, label="TEST F1")

        plt.title(label=f'Качество классификации\n({exp_name})', fontsize=14)
        plt.xlabel(xlabel='Эпоха обучения', fontsize=12)
        plt.ylabel(ylabel='Взвешенная F1-мера', fontsize=12)
        plt.xticks(ticks=np.arange(0, total_epochs + 1, step=5))

        if max(metrics['val_f1']) > 0.5:
            plt.axhline(y=0.80, color='r', linestyle=':', alpha=0.5, label='Целевой уровень (80%)')

        plt.grid(visible=True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right', fontsize=10)

        plt.tight_layout()
        plt.show()

#Сборка модели с нуля
class CustomDogsCNN(nn.Module):
    def __init__(self, num_classes=120):
        super(CustomDogsCNN, self).__init__()

        # Блок извлечения признаков (Свёрточные слои)
        self.features = nn.Sequential(
            # Вход: 3 канала (RGB), 224x224 пикселя
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Размер картинки уменьшается в 2 раза: 112x112

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Размер: 56x56

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Размер: 28x28
        )

        # Блок классификации (Полносвязные слои)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 28 * 28, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Защита от переобучения
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    """
    Главная функция: оркестровка подготовки данных, инициализации параметров и запуска глобальной серии экспериментов.
    """
    # Инициализация вычислительного устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое вычислительное устройство: {device}")

    # Загрузка и подготовка данных (Stanford Dogs Dataset)
    path = kagglehub.dataset_download(handle="jessicali9530/stanford-dogs-dataset")
    print("Путь к локальной копии датасета:", path)
    data_dir = os.path.join(path, 'images/Images')

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset=full_dataset,
        lengths=[train_size, val_size, test_size]
    )

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Устанавливаем количество эпох для глобального эксперимента
    MAX_EPOCHS = 25
    all_results = {}

    print(f"\nЗапуск глобальных экспериментов. Количество эпох: {MAX_EPOCHS}")

    # Эксперимент №1: Pretrained + Adam

    print("\n")
    print(f"ЭКСПЕРИМЕНТ №1: Pretrained (ImageNet) + Adam")

    model_exp1 = create_model(pretrained=True).to(device)
    optimizer_adam = optim.Adam(params=model_exp1.parameters(), lr=1e-4)

    all_results['Exp1_Pretrained_Adam'] = train_model(
        device=device, model=model_exp1, optimizer=optimizer_adam, test_loader=test_loader,
        train_loader=train_loader, train_data=train_data, val_loader=val_loader,
        criterion=criterion, num_epochs=MAX_EPOCHS, use_cutmix=True
    )

    # Эксперимент №2: Pretrained + NAdam

    print("\n")
    print(f"ЭКСПЕРИМЕНТ №2: Pretrained (ImageNet) + NAdam")

    model_exp2 = create_model(pretrained=True).to(device)
    optimizer_nadam = optim.NAdam(params=model_exp2.parameters(), lr=1e-4)

    all_results['Exp2_Pretrained_NAdam'] = train_model(
        device=device, model=model_exp2, optimizer=optimizer_nadam, test_loader=test_loader,
        train_loader=train_loader, train_data=train_data, val_loader=val_loader,
        criterion=criterion, num_epochs=MAX_EPOCHS, use_cutmix=True
    )

    # Эксперимент №3: Обучение базовой архитектуры с нуля (Custom CNN + Adam)

    print("\n")
    print(f"ЭКСПЕРИМЕНТ №3: Custom CNN + Adam (Сборка с нуля)")

    # Используем собранную вручную архитектуру
    model_exp3 = CustomDogsCNN(num_classes=120).to(device)
    optimizer_custom = optim.Adam(params=model_exp3.parameters(), lr=1e-4)

    all_results['Exp3_CustomCNN_Adam'] = train_model(
        device=device, model=model_exp3, optimizer=optimizer_custom, test_loader=test_loader,
        train_loader=train_loader, train_data=train_data, val_loader=val_loader,
        criterion=criterion, num_epochs=MAX_EPOCHS, use_cutmix=True
    )

    print("\nГлобальная серия экспериментов завершена. Формирование графиков...")

    plot_learning_curves(results=all_results, total_epochs=MAX_EPOCHS)


if __name__ == "__main__":
    main()