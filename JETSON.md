# 🧠 Venv com Pacotes Globais (Jetson Nano)

## 📌 Visão Geral

Este ambiente virtual (`venv`) foi configurado para **reutilizar todos os pacotes Python já instalados globalmente no sistema**, evitando reinstalações desnecessárias.

A estratégia combina:

- `--system-site-packages` → acesso direto aos pacotes globais  
- Links simbólicos (`symlinks`) → visibilidade explícita dentro do `venv`  

### ✅ Benefícios

- 🚀 Setup extremamente rápido  
- 💾 Economia significativa de espaço (essencial no Jetson Nano)  
- 🔁 Reutilização automática de bibliotecas pesadas (ex: `torch`, `opencv`, `numpy`)  
- ⚡ Sem necessidade de rebuild ou reinstalação  

---

## 📁 Estrutura do Ambiente

/home/jetson/Desktop/dev/probabilistic-design-dl
├── venv/
├── src/
├── README.md

---

## 📍 Caminhos Importantes

- Projeto: /home/jetson/Desktop/dev/probabilistic-design-dl  
- Venv: venv/  
- Pip global: /usr/local/lib/python3.6/dist-packages  
- Sistema: /usr/lib/python3/dist-packages  

---

## ⚙️ Criação do Ambiente

```bash
deactivate 2>/dev/null
cd /home/jetson/Desktop/dev/probabilistic-design-dl
rm -rf venv
python3 -m venv --system-site-packages venv
source venv/bin/activate
cd venv/lib/python3.6/site-packages
ln -s /usr/local/lib/python3.6/dist-packages/* .
ln -s /usr/lib/python3/dist-packages/* .
cd /home/jetson/Desktop/dev/probabilistic-design-dl
```

---

## ▶️ Uso

```bash
source venv/bin/activate
```

---

## ✅ Teste

```bash
python -c "import numpy, torch, cv2; print('OK')"
```

---

## ⚠️ Observações

- Evita duplicação de pacotes pesados
- Ideal para Jetson Nano
- Atualizações globais são refletidas automaticamente

---

## 👨‍💻 Autor

Mateus Cruz