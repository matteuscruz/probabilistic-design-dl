# Venv com Pacotes Globais (Jetson Nano)

## Descrição
Este venv (`venv`) foi configurado para **ter todos os pacotes Python globais disponíveis** sem precisar reinstalá-los.  
Ele utiliza links simbólicos (`symlinks`) para os diretórios globais de pacotes, garantindo **economia de espaço** e **acesso imediato a todos os pacotes instalados globalmente**.

---

## Caminhos Importantes
- Diretório do projeto: `/home/jetson/Desktop/dev/probabilistic-design-dl`  
- Venv: `venv`  
- Diretórios globais de pacotes:
  - `/usr/local/lib/python3.6/dist-packages` (pip global)
  - `/usr/lib/python3/dist-packages` (pacotes do sistema)

---

## Passo a Passo para Restaurar ou Criar o Venv

1. **Sair de qualquer venv ativo**:
```bash
deactivate
