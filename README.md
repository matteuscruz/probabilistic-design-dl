# probabilistic-design-dl

Projeto com baseline de `naive_bayes` (Iris) e modelos CNN para MNIST:

- `cnn_deterministic`
- `cnn_probabilistic`
- `bayesian_cnn`

## 1) Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Como executar

A entrada principal é `main.py`.

```bash
python main.py
```

Você também pode apontar um arquivo de configuração específico:

```bash
python main.py --config config/naive_bayes.yaml
```

Ou por argumento posicional:

```bash
python main.py config/naive_bayes.yaml
```

> O modelo executado depende de `model.name` no arquivo de configuração.

Config padrão: `config/default.yaml`

Configurações prontas por modelo:

- `config/naive_bayes.yaml`
- `config/cnn_deterministic.yaml`
- `config/cnn_probabilistic.yaml`
- `config/bayesian_cnn.yaml`

---

## 3) Rodar cada modelo

### 3.1 Naive Bayes (Iris)

Execute:

```bash
python main.py --config config/naive_bayes.yaml
```

Saída esperada (exemplo):

```text
Accuracy: 0.7000
```

### 3.2 CNN Determinística (MNIST)

Execute:

```bash
python main.py --config config/cnn_deterministic.yaml
```

### 3.3 CNN Probabilística (MNIST)

Execute:

```bash
python main.py --config config/cnn_probabilistic.yaml
```

### 3.4 Rede Bayesiana CNN (MNIST)

Execute:

```bash
python main.py --config config/bayesian_cnn.yaml
```

---

## 4) Como voltar para Naive Bayes

Basta rodar com o arquivo do Naive Bayes:

```bash
python main.py --config config/naive_bayes.yaml
```

---

## 5) Estrutura de dados para CNNs

Para os modelos CNN, o projeto espera arquivos `.npy` em:

```text
data/
  MNIST/
    x_train.npy
    y_train.npy
    x_test.npy
    y_test.npy
  MNIST_corrupted/
    x_train.npy
    y_train.npy
    x_test.npy
    y_test.npy
```

Esses nomes podem ser alterados no `config/default.yaml`:

```yaml
data:
  root: data
  mnist_name: MNIST
  mnist_corrupted_name: MNIST_corrupted
```

Se os arquivos não existirem, o loader tenta provisionar automaticamente:

- `MNIST`: download via `tf.keras.datasets.mnist`
- `MNIST_corrupted`: gerado automaticamente a partir de `MNIST` com ruído gaussiano

### Formato esperado

- `x_train`, `x_test`: imagens (idealmente `N x 28 x 28` ou `N x 28 x 28 x 1`)
- `y_train`, `y_test`: labels inteiros (`0..9`)

O loader normaliza as imagens para `float32` e adiciona canal automaticamente se necessário.

---

## 6) Ajustar treino

Parâmetros no `config/default.yaml`:

```yaml
train:
  epochs: 500
  learning_rate: 0.01
  seed: 42
  verbose: 0
```

Para CNNs, use valores menores para teste rápido (exemplo):

```yaml
train:
  epochs: 1
  learning_rate: 0.001
  seed: 42
  verbose: 1
```

---

## 7) Artifacts de experimento

A cada execução de pipeline de treino, é criado automaticamente um diretório incremental:

```text
artifacts/
  exp0/
    model/
    history/
    figures/
  exp1/
    ...
```

### Conteúdo salvo

- `model/`
  - Naive Bayes: `naive_bayes.npz`
  - CNNs: `<model_name>.h5`
- `history/`
  - `metadata.json`
  - métricas e históricos em `.csv`
- `figures/`
  - Naive Bayes: plots equivalentes aos usados no notebook
  - CNNs: `training_history.png`

### Configuração

Nos YAMLs em `config/`, você pode ajustar:

```yaml
artifacts:
  enabled: true
  base_dir: artifacts
  save_model: true
  save_history: true
  save_figures: true
  naive_binary_epochs: 50
```

- `naive_binary_epochs` controla o custo das figuras binárias/logísticas do Naive Bayes.

---

## 8) Testes

```bash
python -m pytest -q
```

Com cobertura:

```bash
python -m pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
```

---

## 9) CI

Workflow em `.github/workflows/ci.yml` roda:

- `ruff check .`
- `black --check .`
- `pytest` com cobertura mínima de `70%`

Disparo em:

- `push` (todas as branches)
- `pull_request` para `main`
