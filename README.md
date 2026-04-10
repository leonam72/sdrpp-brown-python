# SDR++Brown Python

> Rebuild completo em Python do [SDR++Brown](https://github.com/sannysanoff/SDRPlusPlusBrown) — receptor SDR multi-VFO com suporte a RTL-SDR V3/V4, espectro FFT, waterfall e demodulação de áudio em tempo real.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![License](https://img.shields.io/badge/License-GPLv3-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## Funcionalidades

- **Multi-VFO** — até 8 VFOs simultâneos, cada um com modo/BW/ganho AF/squelch independente
- **Espectro FFT em tempo real** com waterfall colorido
- **Modos de demodulação:** FM, WFM, AM, USB, LSB, CW, RAW
- **Suporte RTL-SDR V3 e V4** (Rafael Micro R828D) com bias-tee e correção PPM
- **Modo demo** sem hardware (sinal FM sintético)
- **Favoritos** persistentes em JSON
- Presets rápidos: FM BR, Aviação, ISS, PMR446, NOAA, ADS-B
- Interface gráfica nativa via **Tkinter** (sem dependências externas de UI)

---

## Instalação

### 1. Requisitos

- Python 3.10 ou superior
- pip

### 2. Clonar o repositório

```bash
git clone https://github.com/leonam72/sdrpp-brown-python.git
cd sdrpp-brown-python
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
# Obrigatórias
pip install numpy scipy

# Opcional (audio de saída)
pip install sounddevice

# Opcional (hardware RTL-SDR real)
pip install pyrtlsdr
```

> **Windows:** Para usar RTL-SDR real, instale o [driver Zadig](https://zadig.akeo.ie/) e o `rtlsdr.dll` no PATH.

---

## Uso

```bash
python sdrpp_brown.py
```

Se nenhum RTL-SDR estiver conectado, o app inicia automaticamente em **modo demo** com um sinal FM sintético — espectro, waterfall e multi-VFO funcionam imediatamente.

---

## Controles

### Espectro

| Ação | Resultado |
|---|---|
| Clique em área vazia | Move VFO ativo para essa frequência |
| Clicar e arrastar em VFO | Sintoniza aquele VFO |
| Scroll do mouse | Ajusta faixa de dB exibida |

### Painel Multi-VFO

| Controle | Função |
|---|---|
| `● VFO N` | Clique para selecionar o VFO ativo |
| Combo Modo | FM / WFM / AM / USB / LSB / CW / RAW |
| Campo BW | Largura de banda em Hz |
| Slider AF | Ganho de áudio independente |
| Checkbox `🔊` | Mute/ativo por VFO |
| Botão `✕` | Remove o VFO |
| `＋ Novo VFO` | Adiciona um VFO (máx 8) |

### Toolbar

| Campo | Descrição |
|---|---|
| Centro | Frequência central da janela de visualização |
| SR | Sample rate (250k – 3.2M sps) |
| Ganho | Ganho do RTL-SDR (Auto ou 0–50 dB) |
| Min/Max dB | Faixa dinâmica do espectro |

---

## Configuração RTL-SDR V4

Na aba **⚙ SDR**:

- **PPM**: correção de frequência (TCXO)
- **Bias-Tee**: alimentação 4.5V/200mA para LNA externo (V3 e V4)
- **Hardware info**: mostra modelo detectado (R820T2 = V3, R828D = V4)

---

## Arquitetura

```
sdrpp_brown.py
├── VFO              # Objeto por canal: freq, modo, BW, demodulação
├── SDRDevice        # Abstração hardware RTL-SDR + modo demo
├── make_spectrum()  # Canvas FFT + overlay multi-VFO
├── make_waterfall() # Canvas cascata com paleta de cores
├── make_ruler()     # Régua de frequência
├── VFOPanel         # Widget de controle por VFO
└── SDRppBrownApp    # App principal, loop SDR/áudio em threads
```

---

## Dependências

| Pacote | Versão mínima | Descrição |
|---|---|---|
| `numpy` | 1.24.0 | Arrays e FFT |
| `scipy` | 1.10.0 | Filtros FIR/IIR e Hilbert |
| `sounddevice` | 0.4.6 | Saída de áudio (opcional) |
| `pyrtlsdr` | 0.3.0 | Interface RTL-SDR (opcional) |
| `tkinter` | embutido | Interface gráfica (stdlib) |

---

## Baseado em

- [SDR++Brown](https://github.com/sannysanoff/SDRPlusPlusBrown) por sannysanoff (C++/ImGui)
- [SDR++](https://github.com/AlexandreRouma/SDRPlusPlus) por Alexandre Rouma

---

## Licença

GNU General Public License v3.0 — veja [LICENSE](LICENSE).
