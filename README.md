Porównanie efektywności metod zapobiegających przeuczeniu SSN. 

W projekcie 3 zbiorów danych z UC Irvine Machine Learning Repository  https://archive.ics.uci.edu/ml/index.php,
proszę stworzyć kilka sieci neuronowych typu MLP dla zagadnienia klasyfikacji,
które będą charakteryzować się przeuczeniem. Po wykazaniu istnienia takiego zjawiska,
proszę zbadać możliwości zastosowania i efektywność takich metod zapobiegajacych przeuczeniu jak:
Dropout, Regularization L1 i L2, Early Stopping, uproszczenie modelu oraz Data Augmentation.
W trakcie analizy proszę rozważyć różne metody ewaluacji zadania klasyfikacji,
metody detekcji przeuczenia oraz czas niezbędny na wprowadzenie odpowiednich algorytmów.

## How to setup
### It is recommended to update pip before installing any packages
```bash
    python -m pip install --upgrade pip
```
### Install all dependencies
```bash
    python -m pip install -r requirements.txt
```

## How to run
```bash
    python main.py
```

## Datasets
https://archive.ics.uci.edu/dataset/110/yeast

https://archive.ics.uci.edu/dataset/1150/gallstone-1

https://archive.ics.uci.edu/dataset/2/adult

https://archive.ics.uci.edu/dataset/53/iris