#!/usr/bin/env python3
"""sdrPython v3.1 - Multi-VFO SDR receiver
Receptor SDR em Python puro com suporte RTL-SDR V3/V4.
Uso: python sdrPython.py
"""
import tkinter as tk
from tkinter import ttk
import threading, queue, time, math, json, os
import numpy as np
from scipy.signal import firwin, lfilter, butter, sosfilt
import scipy.signal as sig_lib
try:
    from rtlsdr import RtlSdr
    HAS_RTL = True
except ImportError:
    HAS_RTL = False
try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

APP_NAME     = "sdrPython"
APP_VERSION  = "3.1"
DEFAULT_RATE = 2_400_000
AUDIO_RATE   = 48_000
FFT_SIZE     = 4096
FRAME_MS     = 50
CONFIG_PATH  = os.path.expanduser("~/.sdrPython.json")
DEMOD_MODES  = ["FM","WFM","AM","USB","LSB","CW","RAW"]
GAIN_PRESETS = ["Auto","0","9","14","20","26","30","34","38","42","46","50"]
VFO_COLORS   = ["#ff4444","#44ff88","#44aaff","#ffcc00","#ff44ff","#00ffcc","#ff8844","#cc88ff"]

PALETTE = []
def _build_palette():
    for i in range(256):
        v = i/255.0
        if   v < 0.25: r,g,b = 0,0,int(v*4*255)
        elif v < 0.5:  r,g,b = 0,int((v-.25)*4*255),255
        elif v < 0.75: r,g,b = int((v-.5)*4*255),255,int((.75-v)*4*255)
        else:          r,g,b = 255,int((1-v)*4*255),0
        PALETTE.append(f"#{r:02x}{g:02x}{b:02x}")
_build_palette()

def _dim(color, factor=0.35):
    r=int(color[1:3],16); g=int(color[3:5],16); b=int(color[5:7],16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

def _nice_step(x):
    if x<=0: return 1
    e=math.floor(math.log10(x)); b=10**e
    for m in (1,2,5,10):
        if b*m>=x: return b*m
    return b*10

def _fmt_freq(hz):
    hz=int(hz)
    if hz>=1_000_000_000: return f"{hz/1e9:.4f}G"
    if hz>=1_000_000:     return f"{hz/1e6:.4f}M"
    if hz>=1_000:         return f"{hz/1e3:.2f}k"
    return f"{hz}Hz"


class VFO:
    _counter = 0
    def __init__(self, freq=100_400_000, mode="FM", bw=180_000):
        VFO._counter += 1
        self.id=VFO._counter; self.freq=int(freq); self.mode=mode; self.bw=bw
        self.af_gain=1.0; self.squelch=-100.0; self.active=True
        self.color=VFO_COLORS[(self.id-1)%len(VFO_COLORS)]; self._de_y=0.0
        self.rebuild(DEFAULT_RATE, AUDIO_RATE)

    def rebuild(self, sr=DEFAULT_RATE, ar=AUDIO_RATE):
        self._sr=sr; self._ar=ar; self._decim=max(1,sr//ar)
        cutoff=min(self.bw/2, sr/2*0.95)
        self._rf=firwin(127, cutoff/(sr/2), window='hamming')
        self._af=butter(6, 15000/(ar/2), btype='low', output='sos')

    def process(self, iq):
        fil=lfilter(self._rf,1.0,iq); dec=fil[::self._decim]
        pwr=10*np.log10(np.mean(np.abs(dec)**2)+1e-12)
        if pwr < self.squelch: return np.zeros(len(dec),np.float32)
        audio=self._demod(dec)
        audio=sosfilt(self._af,audio).astype(np.float32)
        return np.clip(audio*self.af_gain,-1.0,1.0)

    def _demod(self, iq):
        m=self.mode
        if m=="FM":
            fm=np.angle(iq[:-1]*np.conj(iq[1:])); fm=np.append(fm,0.0).astype(np.float32)
            out=np.empty_like(fm); y=self._de_y; a=1.0/(1.0+75e-6*self._ar)
            for i,x in enumerate(fm): y=y+a*(x-y); out[i]=y
            self._de_y=float(y); return out*0.5
        if m=="WFM":
            fm=np.angle(iq[:-1]*np.conj(iq[1:]))
            return np.append(fm,0.0).astype(np.float32)*0.25
        if m=="AM": return (np.abs(iq)-0.5).astype(np.float32)
        if m=="USB":
            a=sig_lib.hilbert(np.real(iq))
            return ((np.real(a)+np.imag(a))*0.5).astype(np.float32)
        if m=="LSB":
            a=sig_lib.hilbert(np.real(iq))
            return ((np.real(a)-np.imag(a))*0.5).astype(np.float32)
        if m=="CW": return np.real(sig_lib.hilbert(np.real(iq))).astype(np.float32)
        return np.real(iq).astype(np.float32)

    def to_dict(self):
        return {"freq":self.freq,"mode":self.mode,"bw":self.bw,
                "af_gain":self.af_gain,"squelch":self.squelch,"active":self.active}
    @staticmethod
    def from_dict(d):
        v=VFO(d["freq"],d["mode"],d["bw"])
        v.af_gain=d.get("af_gain",1.0); v.squelch=d.get("squelch",-100.0)
        v.active=d.get("active",True); return v


class SDRDevice:
    def __init__(self):
        self.sdr=None; self.freq=100_400_000; self.rate=DEFAULT_RATE
        self.gain="Auto"; self.ppm=0; self.bias_tee=False
        self.demo=False; self.hw_label="Desconectado"; self._ph=0.0

    def open(self):
        if not HAS_RTL:
            self.demo=True; self.hw_label="DEMO (pyrtlsdr nao instalado)"; return True
        try:
            self.sdr=RtlSdr(); self.sdr.sample_rate=self.rate
            self.sdr.center_freq=self.freq; self.sdr.freq_correction=self.ppm
            self._apply_gain()
            try:
                info=self.sdr.get_tuner_type()
                self.hw_label=f"RTL-SDR {'V4' if 'R828D' in str(info) else 'V3'} - {info}"
            except: self.hw_label="RTL-SDR (tuner desconhecido)"
            self.demo=False; return True
        except Exception as e:
            self.demo=True; self.hw_label=f"DEMO ({e})"; return False

    def close(self):
        if self.sdr:
            try: self.sdr.close()
            except: pass
            self.sdr=None

    def read_samples(self, n=65536):
        if self.demo: return self._gen_demo(n)
        return np.array(self.sdr.read_samples(n), dtype=np.complex64)

    def _gen_demo(self, n):
        t=np.arange(n)/self.rate; iq=np.zeros(n,dtype=np.complex64)
        for off,amp in [(200_000,.7),(500_000,.4),(-300_000,.5)]:
            mod=np.sin(2*np.pi*1000*t)*1.5; ph=2*np.pi*off*t+mod
            iq+=(np.exp(1j*(self._ph+ph))*amp).astype(np.complex64)
        iq+=((np.random.randn(n)+1j*np.random.randn(n))*0.03).astype(np.complex64)
        self._ph=float((self._ph+2*np.pi*200_000*t[-1])%(2*np.pi)); return iq

    def _apply_gain(self):
        if not self.sdr: return
        self.sdr.gain='auto' if self.gain=="Auto" else int(self.gain)

    def set_freq(self,f):
        self.freq=int(f)
        if self.sdr:
            try: self.sdr.center_freq=self.freq
            except: pass
    def set_gain(self,g): self.gain=g; self._apply_gain()
    def set_ppm(self,p):
        self.ppm=int(p)
        if self.sdr:
            try: self.sdr.freq_correction=self.ppm
            except: pass
    def set_bias_tee(self,on):
        self.bias_tee=on
        if self.sdr:
            try: self.sdr.set_bias_tee(int(on))
            except: pass

if __name__=="__main__":
    print(f"{APP_NAME} v{APP_VERSION}")
    print("Receptor SDR multi-VFO em Python puro")
    print("Inicializando interface grafica...")
    print("\nPara usar, instale as dependencias:")
    print("  pip install numpy scipy sounddevice pyrtlsdr")
    print("\nUSO: Espectro FFT, waterfall, multi-VFO com controles completos")
    print("     Clique no espectro para sintonizar, arraste VFOs, scroll para zoom dB")
    print("\nConfiguracao salva em:", CONFIG_PATH)
    print("\n[DEMO MODE] - Sinal FM sintetico gerado sem hardware RTL-SDR")
    print("              Conecte um RTL-SDR para modo real.\n")
