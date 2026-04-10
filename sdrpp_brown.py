#!/usr/bin/env python3
"""
SDR++Brown Python — v3.1
Multi-VFO: até 8 VFOs simultâneos, cada um com modo/BW/AF independente.
Suporte: RTL-SDR V4 (R828D), V3, modo demo.
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

DEFAULT_RATE = 2_400_000
AUDIO_RATE   = 48_000
FFT_SIZE     = 4096
FRAME_MS     = 50
CONFIG_PATH  = os.path.expanduser("~/.sdrpp_brown_v3.json")
DEMOD_MODES  = ["FM","WFM","AM","USB","LSB","CW","RAW"]
GAIN_PRESETS = ["Auto","0","9","14","20","26","30","34","38","42","46","50"]

VFO_COLORS = ["#ff4444","#44ff88","#44aaff","#ffcc00",
              "#ff44ff","#00ffcc","#ff8844","#cc88ff"]

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
    r = int(color[1:3],16); g = int(color[3:5],16); b = int(color[5:7],16)
    r = int(r*factor); g = int(g*factor); b = int(b*factor)
    return f"#{r:02x}{g:02x}{b:02x}"


class VFO:
    _counter = 0

    def __init__(self, freq=100_400_000, mode="FM", bw=180_000):
        VFO._counter += 1
        self.id      = VFO._counter
        self.freq    = int(freq)
        self.mode    = mode
        self.bw      = bw
        self.af_gain = 1.0
        self.squelch = -100.0
        self.active  = True
        self.color   = VFO_COLORS[(self.id-1) % len(VFO_COLORS)]
        self._de_y   = 0.0
        self.rebuild(DEFAULT_RATE, AUDIO_RATE)

    def rebuild(self, sr=DEFAULT_RATE, ar=AUDIO_RATE):
        self._sr = sr; self._ar = ar
        self._decim = max(1, sr//ar)
        cutoff = min(self.bw/2, sr/2*0.95)
        self._rf  = firwin(127, cutoff/(sr/2), window='hamming')
        self._af  = butter(6, 15000/(ar/2), btype='low', output='sos')

    def process(self, iq):
        fil = lfilter(self._rf, 1.0, iq)
        dec = fil[::self._decim]
        pwr = 10*np.log10(np.mean(np.abs(dec)**2)+1e-12)
        if pwr < self.squelch:
            return np.zeros(len(dec), np.float32)
        audio = self._demod(dec)
        audio = sosfilt(self._af, audio).astype(np.float32)
        return np.clip(audio * self.af_gain, -1.0, 1.0)

    def _demod(self, iq):
        m = self.mode
        if m == "FM":
            fm = np.angle(iq[:-1]*np.conj(iq[1:]))
            fm = np.append(fm, 0.0).astype(np.float32)
            out = np.empty_like(fm); y = self._de_y
            a = 1.0/(1.0+75e-6*self._ar)
            for i,x in enumerate(fm): y = y+a*(x-y); out[i]=y
            self._de_y = float(y)
            return out*0.5
        if m == "WFM":
            fm = np.angle(iq[:-1]*np.conj(iq[1:]))
            return np.append(fm, 0.0).astype(np.float32)*0.25
        if m == "AM":
            return (np.abs(iq)-0.5).astype(np.float32)
        if m == "USB":
            a = sig_lib.hilbert(np.real(iq))
            return ((np.real(a)+np.imag(a))*0.5).astype(np.float32)
        if m == "LSB":
            a = sig_lib.hilbert(np.real(iq))
            return ((np.real(a)-np.imag(a))*0.5).astype(np.float32)
        if m == "CW":
            return np.real(sig_lib.hilbert(np.real(iq))).astype(np.float32)
        return np.real(iq).astype(np.float32)

    def to_dict(self):
        return {"freq":self.freq,"mode":self.mode,"bw":self.bw,
                "af_gain":self.af_gain,"squelch":self.squelch,"active":self.active}

    @staticmethod
    def from_dict(d):
        v = VFO(d["freq"], d["mode"], d["bw"])
        v.af_gain = d.get("af_gain",1.0)
        v.squelch = d.get("squelch",-100.0)
        v.active  = d.get("active",True)
        return v


class SDRDevice:
    def __init__(self):
        self.sdr=None; self.freq=100_400_000; self.rate=DEFAULT_RATE
        self.gain="Auto"; self.ppm=0; self.bias_tee=False
        self.demo=False; self.hw_label="Desconectado"; self._ph=0.0

    def open(self):
        if not HAS_RTL:
            self.demo=True; self.hw_label="DEMO (pyrtlsdr não instalado)"; return True
        try:
            self.sdr=RtlSdr(); self.sdr.sample_rate=self.rate
            self.sdr.center_freq=self.freq; self.sdr.freq_correction=self.ppm
            self._apply_gain()
            try:
                info=self.sdr.get_tuner_type()
                self.hw_label=f"RTL-SDR {'V4' if 'R828D' in str(info) else 'V3'} — {info}"
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
        t = np.arange(n)/self.rate
        iq = np.zeros(n, dtype=np.complex64)
        for off,amp in [(200_000,.7),(500_000,.4),(-300_000,.5)]:
            mod = np.sin(2*np.pi*1000*t)*1.5
            ph  = 2*np.pi*off*t+mod
            iq += (np.exp(1j*(self._ph+ph))*amp).astype(np.complex64)
        iq += ((np.random.randn(n)+1j*np.random.randn(n))*0.03).astype(np.complex64)
        self._ph = float((self._ph+2*np.pi*200_000*t[-1])%(2*np.pi))
        return iq

    def _apply_gain(self):
        if not self.sdr: return
        self.sdr.gain = 'auto' if self.gain=="Auto" else int(self.gain)

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


def make_spectrum(parent, get_vfos, get_active_id, set_active_id,
                 set_vfo_freq, get_center, get_rate):
    c = tk.Canvas(parent, bg='#0a0a0a', highlightthickness=0, height=190)
    c._vmin=-120.; c._vmax=-20.; c._fft=None
    c._drag_x=None; c._drag_f=None; c._drag_vid=None

    def _x_of_freq(f):
        w = c.winfo_width()
        if w < 2: return 0
        return (f - get_center() + get_rate()/2) / get_rate() * w

    def _freq_at_x(x):
        w = c.winfo_width()
        if w < 2: return get_center()
        return get_center() - get_rate()/2 + x/w*get_rate()

    def draw(fft_db):
        c._fft = fft_db
        _redraw()

    def _redraw():
        fft = c._fft
        if fft is None: return
        w = c.winfo_width(); h = c.winfo_height()
        if w < 2 or h < 2: return
        c.delete("all")

        for db in range(int(c._vmin), int(c._vmax)+1, 10):
            y = int((db - c._vmax) / (c._vmin - c._vmax) * h)
            if 0 <= y <= h:
                c.create_line(0, y, w, y, fill='#1c2c2c', dash=(3,5))
                c.create_text(3, y-7, text=f"{db}",
                              fill='#3a6060', anchor='nw', font=('Consolas',7))

        rs = np.interp(np.linspace(0, len(fft)-1, w),
                       np.arange(len(fft)), fft)
        ys = np.clip((rs - c._vmax)/(c._vmin - c._vmax)*h, 0, h).astype(int)
        pts = []
        for x in range(w): pts += [x, int(ys[x])]
        if len(pts) >= 4:
            c.create_polygon([0,h]+pts+[w,h], fill='#002030', outline='')
            c.create_line(pts, fill='#00d4ff', width=1)

        active_id = get_active_id()
        vfos = get_vfos()
        for vfo in vfos:
            xv    = _x_of_freq(vfo.freq)
            col   = vfo.color
            dim   = _dim(col, 0.30)
            bw_px = vfo.bw / get_rate() * w / 2

            if bw_px > 1:
                c.create_rectangle(xv-bw_px, 0, xv+bw_px, h,
                                   fill=dim, outline=col,
                                   dash=(4,4), width=1,
                                   stipple='gray12')

            lw = 2 if vfo.id == active_id else 1
            c.create_line(xv, 0, xv, h, fill=col, width=lw)

            c.create_polygon(xv-7, 0, xv+7, 0, xv, 13, fill=col, outline='')

            label = f"V{vfo.id} {_fmt_freq(vfo.freq)}  {vfo.mode}"
            c.create_rectangle(xv+4, 14, xv+6+len(label)*6, 26,
                                fill='#0a0a0a', outline='')
            c.create_text(xv+5, 15, text=label, fill=col,
                          anchor='nw', font=('Consolas',8,'bold'))

    def _nearest_vfo(x):
        best=None; best_d=float('inf')
        for vfo in get_vfos():
            d = abs(_x_of_freq(vfo.freq) - x)
            bw_px = vfo.bw/get_rate()*c.winfo_width()/2
            if d < max(bw_px+8, 14) and d < best_d:
                best=vfo; best_d=d
        return best

    def on_click(e):
        hit = _nearest_vfo(e.x)
        if hit:
            set_active_id(hit.id)
            c._drag_vid=hit.id; c._drag_x=e.x; c._drag_f=hit.freq
        else:
            vfos = get_vfos()
            if not vfos: return
            aid = get_active_id()
            v   = next((v for v in vfos if v.id==aid), vfos[0])
            set_active_id(v.id)
            nf  = int(_freq_at_x(e.x))
            set_vfo_freq(v.id, nf)
            c._drag_vid=v.id; c._drag_x=e.x; c._drag_f=nf

    def on_drag(e):
        if c._drag_x is None: return
        dx = e.x - c._drag_x
        df = dx / max(c.winfo_width(),1) * get_rate()
        set_vfo_freq(c._drag_vid, int(c._drag_f + df))

    def on_release(e):
        c._drag_x=None; c._drag_f=None; c._drag_vid=None

    def on_scroll(e):
        d = -1 if (e.delta>0 or e.num==4) else 1
        c._vmin = max(-200., min(c._vmax-20, c._vmin+d*5))
        _redraw()

    c.bind("<ButtonPress-1>",   on_click)
    c.bind("<B1-Motion>",       on_drag)
    c.bind("<ButtonRelease-1>", on_release)
    c.bind("<MouseWheel>",      on_scroll)
    c.bind("<Button-4>",        on_scroll)
    c.bind("<Button-5>",        on_scroll)
    c.bind("<Configure>",       lambda _: _redraw())

    def set_range(mn, mx):
        c._vmin=mn; c._vmax=mx; _redraw()

    c.update_fft = draw
    c.set_range  = set_range
    c.redraw     = _redraw
    return c


def make_waterfall(parent):
    c = tk.Canvas(parent, bg='black', highlightthickness=0)
    c._vmin=-120.; c._vmax=-40.; c._rows=[]; c._photo=None
    c._cw=1; c._ch=1
    MAX=200

    def on_cfg(e):
        c._cw=max(1,e.width); c._ch=max(1,e.height); c._rows.clear()
    c.bind("<Configure>", on_cfg)

    def push(fft_db):
        w = c._cw
        if w < 2: return
        rs   = np.interp(np.linspace(0,len(fft_db)-1,w),
                         np.arange(len(fft_db)), fft_db)
        norm = np.clip((rs-c._vmin)/(c._vmax-c._vmin), 0, 1)
        idx  = (norm*255).astype(int)
        row  = "{" + " ".join(PALETTE[i] for i in idx) + "}"
        c._rows.insert(0, row)
        if len(c._rows) > MAX: c._rows.pop()
        h = min(len(c._rows), c._ch)
        if h < 1: return
        try:
            img = tk.PhotoImage(width=w, height=h)
            img.put(" ".join(c._rows[:h]))
            c.delete("all")
            c.create_image(0, 0, anchor='nw', image=img)
            c._photo = img
        except Exception: pass

    c.push_fft  = push
    c.set_range = lambda mn,mx: (setattr(c,'_vmin',mn), setattr(c,'_vmax',mx))
    return c


def make_ruler(parent, get_center, get_rate):
    c = tk.Canvas(parent, bg='#0d1117', height=24, highlightthickness=0)

    def draw():
        w = c.winfo_width()
        if w < 2: return
        c.delete("all")
        center=get_center(); rate=get_rate()
        start=center-rate/2; span=rate
        step=_nice_step(span/7)
        f=math.ceil(start/step)*step
        while f <= start+span:
            x = int((f-start)/span*w)
            if 0 <= x <= w:
                c.create_line(x, 0, x, 10, fill='#2a3a3a')
                c.create_text(x, 11, text=_fmt_freq(f),
                              fill='#5a7070', font=('Consolas',7), anchor='n')
            f += step
        c.create_line(w//2, 0, w//2, 8, fill='#446688', width=2)

    c.bind("<Configure>", lambda _: draw())
    c.redraw = draw
    return c

def _nice_step(x):
    if x<=0: return 1
    e=math.floor(math.log10(x)); b=10**e
    for m in(1,2,5,10):
        if b*m>=x: return b*m
    return b*10

def _fmt_freq(hz):
    hz=int(hz)
    if hz>=1_000_000_000: return f"{hz/1e9:.4f}G"
    if hz>=1_000_000:     return f"{hz/1e6:.4f}M"
    if hz>=1_000:         return f"{hz/1e3:.2f}k"
    return f"{hz}Hz"

if __name__=="__main__":
    print("SDR++Brown Python v3.1")
    print("Para executar o app completo, use o arquivo completo fornecido.")
    print("Este arquivo parcial contém:")
    print("  - VFO (classe)")
    print("  - SDRDevice (abstração RTL-SDR + demo)")
    print("  - make_spectrum, make_waterfall, make_ruler")
    print("\nFalta adicionar: VFOPanel, BookmarkManager, SDRppBrownApp, main loop")
