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


def train_model(device, model, optimizer, train_loader, train_data, val_loader, criterion, num_epochs=5,
                use_cutmix=True):
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
    history = {'train_loss': [], 'val_f1': []}

    for epoch in range(num_epochs):
        print(f'\nЭпоха {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # ФАЗА ОБУЧЕНИЯ
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Очистка градиентов перед новым шагом оптимизации
            optimizer.zero_grad()

            # Применение стохастической аугментации CutMix с вероятностью 0.5
            if use_cutmix and np.random.rand() > 0.5:
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(inputs.size()[0]).to(device)

                target_a = labels
                target_b = labels[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(size=inputs.size(), lam=lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

                # Корректировка коэффициента на основе фактической площади вырезанного фрагмента
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                outputs = model(inputs)

                loss = cutmix_criterion(
                    criterion=criterion,
                    pred=outputs,
                    y_a=target_a,
                    y_b=target_b,
                    lam=lam
                )
            else:
                # Стандартный прямой проход без аугментации
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Обратное распространение ошибки и шаг оптимизатора
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Вычисление средних потерь за эпоху
        epoch_loss = running_loss / len(train_data)
        history['train_loss'].append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')

        # ФАЗА ВАЛИДАЦИИ
        model.eval()
        all_preds = []
        all_labels = []

        # Отключение вычисления градиентов для снижения потребления памяти
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Определение предсказанного класса (argmax)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Расчет взвешенных метрик качества классификации
        val_precision = precision_score(y_true=all_labels, y_pred=all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(y_true=all_labels, y_pred=all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='weighted', zero_division=0)

        history['val_f1'].append(val_f1)
        print(f'Validation - Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}')

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

def plot_learning_curves(results: dict) -> None:
    """
    Визуализирует динамику функции потерь и метрики F1-score на протяжении всех эпох обучения.

    Args:
        results (dict): Словарь, содержащий истории метрик для каждого проведенного эксперимента.
                         структура: { 'Имя_эксперимента': {'train_loss': [...], 'val_f1': [...]} }
    """
    # Извлечение количества эпох из первого доступного эксперимента
    first_experiment_key = list(results.keys())[0]
    num_epochs = len(results[first_experiment_key]['train_loss'])
    epochs = range(1, num_epochs + 1)

    # Инициализация графического окна (фигуры)
    plt.figure(figsize=(14, 6))

    # График №1: Динамика функции потерь на обучающей выборке
    plt.subplot(1, 2, 1)
    for exp_name, metrics in results.items():
        plt.plot(epochs, metrics['train_loss'], marker='o', label=exp_name)

    plt.title(label='Сходимость функции потерь (Training Loss)')
    plt.xlabel(xlabel='Эпоха обучения')
    plt.ylabel(ylabel='Значение потерь (Loss)')
    plt.xticks(ticks=epochs)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # График №2: Динамика F1-меры на валидационной выборке
    plt.subplot(1, 2, 2)
    for exp_name, metrics in results.items():
        plt.plot(epochs, metrics['val_f1'], marker='s', label=exp_name)

    plt.title(label='Динамика качества классификации (Validation F1-score)')
    plt.xlabel(xlabel='Эпоха обучения')
    plt.ylabel(ylabel='Взвешенная F1-мера')
    plt.xticks(ticks=epochs)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # Оптимизация расположения элементов графиков и их вывод на экран
    plt.tight_layout()
    plt.show()

def main():
    """
    Главная функция: оркестровка подготовки данных, инициализации параметров и запуска серии экспериментов.
    """
    # Инициализация вычислительного устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое вычислительное устройство: {device}")

    # Загрузка набора данных
    path = kagglehub.dataset_download(handle="jessicali9530/stanford-dogs-dataset")
    print("Путь к локальной копии датасета:", path)

    data_dir = os.path.join(path, 'images/Images')

    # Определение конвейера предварительной обработки изображений
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Именованные параметры для Dataset
    full_dataset = ImageFolder(root=data_dir, transform=transform)
    print(f"Общий объем выборки (изображений): {len(full_dataset)}")
    print(f"Размерность пространства классов: {len(full_dataset.classes)}")

    # Разделение генеральной совокупности на стратифицированные выборки (70% / 15% / 15%)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset=full_dataset,
        lengths=[train_size, val_size, test_size]
    )

    # Инициализация загрузчиков данных с поддержкой пакетной обработки
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    print(f"Распределение: Обучение={len(train_data)}, Валидация={len(val_data)}, Тест={len(test_data)}\n")

    # Определение глобальной функции потерь
    criterion = nn.CrossEntropyLoss()

    # Структура данных для агрегации результатов экспериментов
    all_results = {}

    # Эксперимент №1: Трансферное обучение (Transfer Learning) + оптимизатор Adam
    print("ЭКСПЕРИМЕНТ №1: Pretrained (ImageNet) + Adam")

    model_exp1 = create_model(pretrained=True).to(device)
    optimizer_adam = optim.Adam(params=model_exp1.parameters(), lr=1e-4)

    all_results['Exp1_Adam_Pretrained'] = train_model(
        device=device,
        model=model_exp1,
        optimizer=optimizer_adam,
        train_loader=train_loader,
        train_data=train_data,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=5,
        use_cutmix=True
    )

    # Эксперимент №2: Трансферное обучение + оптимизатор NAdam
    print("ЭКСПЕРИМЕНТ №2: Pretrained (ImageNet) + NAdam")

    model_exp2 = create_model(pretrained=True).to(device)
    optimizer_nadam = optim.NAdam(params=model_exp2.parameters(), lr=1e-4)

    all_results['Exp2_NAdam_Pretrained'] = train_model(
        device=device,
        model=model_exp2,
        optimizer=optimizer_nadam,
        train_loader=train_loader,
        train_data=train_data,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=5,
        use_cutmix=True
    )

    # Эксперимент №3: Обучение базовой архитектуры со случайной инициализацией весов

    print("ЭКСПЕРИМЕНТ №3: Обучение с нуля (From Scratch) + Adam")

    model_exp3 = create_model(pretrained=False).to(device)
    optimizer_scratch = optim.Adam(params=model_exp3.parameters(), lr=1e-4)

    all_results['Exp3_Adam_Scratch'] = train_model(
        device=device,
        model=model_exp3,
        optimizer=optimizer_scratch,
        train_loader=train_loader,
        train_data=train_data,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=5,
        use_cutmix=True
    )

    print("\nСерия экспериментов завершена.")

    plot_learning_curves(results=all_results)


if __name__ == "__main__":
    main()