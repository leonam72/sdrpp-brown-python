#!/usr/bin/env python3
"""
SDR++Brown Python  — v3.0
Multi-VFO: até 8 VFOs simultâneos, cada um com modo/BW/ganho AF independente.
Suporte: RTL-SDR V4 (R828D), V3, modo demo.
"""

import tkinter as tk
from tkinter import ttk
import threading, queue, time, math, json, os, colorsys
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

# Cores distintas para cada VFO
VFO_COLORS = [
    "#ff4444",  # 0 vermelho
    "#44ff88",  # 1 verde
    "#44aaff",  # 2 azul
    "#ffcc00",  # 3 amarelo
    "#ff44ff",  # 4 magenta
    "#00ffcc",  # 5 ciano
    "#ff8844",  # 6 laranja
    "#cc88ff",  # 7 roxo
]

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


# ─────────────────────────────────────────────────────────────────────────────
#  VFO  — um canal de recepção independente
# ─────────────────────────────────────────────────────────────────────────────
class VFO:
    _id_counter = 0

    def __init__(self, freq=100_400_000, mode="FM", bw=180_000):
        VFO._id_counter += 1
        self.id       = VFO._id_counter
        self.freq     = int(freq)
        self.mode     = mode
        self.bw       = bw
        self.af_gain  = 1.0
        self.squelch  = -100.0
        self.active   = True       # produz áudio/demodulação
        self.color    = VFO_COLORS[(self.id - 1) % len(VFO_COLORS)]
        # estado DSP interno
        self._rf_taps = None
        self._af_sos  = None
        self._de_y    = 0.0
        self._build_filters(DEFAULT_RATE, AUDIO_RATE)

    def _build_filters(self, sr, ar):
        self._sr   = sr
        self._ar   = ar
        self._decim = max(1, sr // ar)
        cutoff = min(self.bw/2, sr/2*0.95)
        self._rf_taps = firwin(127, cutoff/(sr/2), window='hamming')
        self._af_sos  = butter(6, 15000/(ar/2), btype='low', output='sos')

    def rebuild(self, sr=None, ar=None):
        self._build_filters(sr or self._sr, ar or self._ar)

    def process(self, iq: np.ndarray):
        """Retorna (fft_db ignorado, audio float32) ou None se squelch."""
        fil = lfilter(self._rf_taps, 1.0, iq)
        dec = fil[::self._decim]
        pwr = 10*np.log10(np.mean(np.abs(dec)**2)+1e-12)
        if pwr < self.squelch:
            return np.zeros(len(dec), np.float32)
        audio = self._demod(dec)
        audio = sosfilt(self._af_sos, audio).astype(np.float32)
        return np.clip(audio * self.af_gain, -1.0, 1.0)

    def _demod(self, iq):
        m = self.mode
        if m == "FM":
            fm = np.angle(iq[:-1]*np.conj(iq[1:]))
            fm = np.append(fm,0.0).astype(np.float32)
            out = np.zeros_like(fm); y = self._de_y
            a   = 1/(1 + 75e-6*self._ar)
            for i,x in enumerate(fm): y = y+a*(x-y); out[i]=y
            self._de_y = float(y)
            return out*0.5
        if m == "WFM":
            fm = np.angle(iq[:-1]*np.conj(iq[1:]))
            return np.append(fm,0.0).astype(np.float32)*0.25
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


# ─────────────────────────────────────────────────────────────────────────────
#  SDR Device
# ─────────────────────────────────────────────────────────────────────────────
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

    def read_samples(self,n=65536):
        if self.demo: return self._gen_demo(n)
        return np.array(self.sdr.read_samples(n),dtype=np.complex64)

    def _gen_demo(self,n):
        t=np.arange(n)/self.rate
        # gera 3 sinais FM sintéticos em offsets diferentes
        iq = np.zeros(n,dtype=np.complex64)
        for off,amp in [(200_000,.7),(500_000,.4),(-300_000,.5)]:
            mod=np.sin(2*np.pi*1000*t)*1.5
            ph=2*np.pi*off*t+mod
            iq+=(np.exp(1j*(self._ph+ph))*amp).astype(np.complex64)
        iq+=((np.random.randn(n)+1j*np.random.randn(n))*0.03).astype(np.complex64)
        self._ph=float((self._ph+2*np.pi*200_000*t[-1])%(2*np.pi))
        return iq

    def _apply_gain(self):
        if not self.sdr: return
        self.sdr.gain='auto' if self.gain=="Auto" else int(self.gain)

    def set_freq(self,f): self.freq=int(f);
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


# ─────────────────────────────────────────────────────────────────────────────
#  Spectrum canvas com multi-VFO
# ─────────────────────────────────────────────────────────────────────────────
def make_spectrum(parent, get_vfos, get_active_id, set_active_id,
                 set_vfo_freq, get_center, get_rate):
    c = tk.Canvas(parent, bg='#0a0a0a', highlightthickness=0, height=190)
    c._vmin=-120.; c._vmax=-20.; c._fft=None
    c._drag_x=None; c._drag_f=None; c._drag_vid=None

    def _freq_at_x(x):
        w=c.winfo_width(); rate=get_rate()
        return get_center()-rate/2+x/w*rate if w>1 else get_center()

    def _x_of_freq(f):
        w=c.winfo_width(); rate=get_rate()
        return (f-get_center()+rate/2)/rate*w

    def draw(fft_db):
        c._fft=fft_db; _redraw()

    def _redraw():
        fft=c._fft
        if fft is None: return
        w=c.winfo_width(); h=c.winfo_height()
        if w<2 or h<2: return
        c.delete("all")
        # Grid
        for db in range(int(c._vmin),int(c._vmax)+1,10):
            y=int((db-c._vmax)/(c._vmin-c._vmax)*h)
            if 0<=y<=h:
                c.create_line(0,y,w,y,fill='#1c2c2c',dash=(3,5))
                c.create_text(3,y-7,text=f"{db}",fill='#3a6060',
                              anchor='nw',font=('Consolas',7))
        # Espectro
        rs=np.interp(np.linspace(0,len(fft)-1,w),np.arange(len(fft)),fft)
        ys=np.clip((rs-c._vmax)/(c._vmin-c._vmax)*h,0,h).astype(int)
        pts=[]
        for x in range(w): pts+=[x,int(ys[x])]
        if len(pts)>=4:
            c.create_polygon([0,h]+pts+[w,h],fill='#002030',outline='')
            c.create_line(pts,fill='#00d4ff',width=1)

        # Cada VFO
        active_id = get_active_id()
        for vfo in get_vfos():
            xv = _x_of_freq(vfo.freq)
            col = vfo.color
            bw_px = vfo.bw/get_rate()*w/2
            # sombra BW
            alpha_fill = col+'22'  # transparente
            c.create_rectangle(xv-bw_px,0,xv+bw_px,h,
                                fill='',outline=col+'44',
                                dash=(2,4))
            # fill leve
            c.create_rectangle(xv-bw_px,0,xv+bw_px,h,
                                fill=col+'11',outline='',stipple='gray12')
            # linha vertical
            lw = 3 if vfo.id==active_id else 1
            c.create_line(xv,0,xv,h,fill=col,width=lw)
            # triângulo no topo
            c.create_polygon(xv-7,0,xv+7,0,xv,13,fill=col)
            # label
            label = f"VFO{vfo.id} {_fmt_freq(vfo.freq)}"
            c.create_text(xv+5,15,text=label,fill=col,
                          anchor='nw',font=('Consolas',8,'bold'))

    # ── Mouse ──
    def _vfo_at_x(x):
        """Retorna VFO mais próximo do clique (se dentro da BW)."""
        f = _freq_at_x(x)
        w = c.winfo_width(); rate = get_rate()
        best_v = None; best_d = float('inf')
        for vfo in get_vfos():
            xv = _x_of_freq(vfo.freq)
            d  = abs(xv - x)
            bw_px = vfo.bw/rate*w/2
            if d < max(bw_px+6, 12) and d < best_d:
                best_d = d; best_v = vfo
        return best_v

    def on_click(e):
        hit = _vfo_at_x(e.x)
        if hit:
            set_active_id(hit.id)
            c._drag_vid = hit.id
            c._drag_x   = e.x
            c._drag_f   = hit.freq
        else:
            # sem VFO próximo → move VFO ativo
            active_id = get_active_id()
            vfos = get_vfos()
            if vfos:
                v = next((v for v in vfos if v.id==active_id), vfos[0])
                set_active_id(v.id)
                c._drag_vid = v.id
                c._drag_x   = e.x
                c._drag_f   = _freq_at_x(e.x)
                set_vfo_freq(v.id, int(c._drag_f))

    def on_drag(e):
        if c._drag_x is None: return
        dx = e.x - c._drag_x
        rate = get_rate(); w2 = c.winfo_width()
        df = dx/w2*rate
        new_f = int(c._drag_f + df)
        set_vfo_freq(c._drag_vid, new_f)

    def on_release(e): c._drag_x=None; c._drag_f=None; c._drag_vid=None

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

    c.update_fft = draw
    c.set_range  = lambda mn,mx: (setattr(c,'_vmin',mn), setattr(c,'_vmax',mx), _redraw())
    c.redraw     = _redraw
    return c


# ─────────────────────────────────────────────────────────────────────────────
#  Waterfall
# ─────────────────────────────────────────────────────────────────────────────
def make_waterfall(parent):
    c=tk.Canvas(parent,bg='black',highlightthickness=0)
    c._vmin=-120.; c._vmax=-40.; c._rows=[]; c._photo=None; c._cw=1; c._ch=1
    MAX=200
    def on_cfg(e): c._cw=max(1,e.width); c._ch=max(1,e.height); c._rows.clear()
    c.bind("<Configure>",on_cfg)
    def push(fft_db):
        w=c._cw
        if w<2: return
        rs  =np.interp(np.linspace(0,len(fft_db)-1,w),np.arange(len(fft_db)),fft_db)
        norm=np.clip((rs-c._vmin)/(c._vmax-c._vmin),0,1)
        idx =(norm*255).astype(int)
        row ="{"+' '.join(PALETTE[i] for i in idx)+"}"
        c._rows.insert(0,row)
        if len(c._rows)>MAX: c._rows.pop()
        h=min(len(c._rows),c._ch)
        if h<1: return
        try:
            img=tk.PhotoImage(width=w,height=h)
            img.put(' '.join(c._rows[:h]))
            c.delete('all'); c.create_image(0,0,anchor='nw',image=img); c._photo=img
        except: pass
    c.push_fft=push
    c.set_range=lambda mn,mx: (setattr(c,'_vmin',mn),setattr(c,'_vmax',mx))
    return c


# ─────────────────────────────────────────────────────────────────────────────
#  Régua
# ─────────────────────────────────────────────────────────────────────────────
def make_ruler(parent, get_center, get_rate):
    c=tk.Canvas(parent,bg='#0d1117',height=24,highlightthickness=0)
    def draw():
        w=c.winfo_width()
        if w<2: return
        c.delete("all")
        center=get_center(); rate=get_rate()
        start=center-rate/2; stop=center+rate/2; span=stop-start
        step=_nice_step(span/7)
        f=math.ceil(start/step)*step
        while f<=stop:
            x=int((f-start)/span*w)
            c.create_line(x,0,x,10,fill='#2a3a3a')
            c.create_text(x,11,text=_fmt_freq(f),fill='#5a7070',
                          font=('Consolas',7),anchor='n')
            f+=step
        c.create_line(w//2,0,w//2,8,fill='#446688',width=2)
    c.bind("<Configure>",lambda _:draw())
    c.redraw=draw; return c


# ─────────────────────────────────────────────────────────────────────────────
#  Bookmark Manager
# ─────────────────────────────────────────────────────────────────────────────
class BookmarkManager:
    def __init__(self): self.items=[]
    def add(self,name,freq,mode,bw): self.items.append({"name":name,"freq":freq,"mode":mode,"bw":bw})
    def remove(self,i):
        if 0<=i<len(self.items): del self.items[i]
    def to_list(self): return self.items
    def from_list(self,d): self.items=list(d)


# ─────────────────────────────────────────────────────────────────────────────
#  VFO Panel (widget lateral)
# ─────────────────────────────────────────────────────────────────────────────
class VFOPanel(tk.Frame):
    """Widget que representa um VFO na lista lateral."""
    def __init__(self, parent, vfo: VFO, on_select, on_remove, on_change, **kw):
        super().__init__(parent, bg='#161b22',
                         highlightthickness=1,
                         highlightbackground=vfo.color, **kw)
        self.vfo = vfo
        self._on_select = on_select
        self._on_remove = on_remove
        self._on_change = on_change
        self._build()
        self.bind("<Button-1>", lambda _: on_select(vfo.id))

    def _build(self):
        v = self.vfo
        top = tk.Frame(self, bg='#161b22')
        top.pack(fill='x', padx=4, pady=(3,0))

        # Indicador de cor + ID
        tk.Label(top, text=f"● VFO {v.id}", bg='#161b22', fg=v.color,
                 font=('Consolas',9,'bold')).pack(side='left')

        # Ativo/mudo
        self._active_v = tk.BooleanVar(value=v.active)
        tk.Checkbutton(top, text="Áudio", variable=self._active_v,
                       bg='#161b22', fg='#8b949e', selectcolor='#21262d',
                       activebackground='#161b22',
                       command=self._toggle_active).pack(side='left', padx=6)

        tk.Button(top, text="✕", bg='#161b22', fg='#f85149',
                  relief='flat', font=('Consolas',9),
                  command=lambda: self._on_remove(v.id)).pack(side='right', padx=2)

        mid = tk.Frame(self, bg='#161b22')
        mid.pack(fill='x', padx=4, pady=2)

        # Frequência
        tk.Label(mid, text="Hz:", bg='#161b22', fg='#6e7681',
                 font=('Consolas',8)).pack(side='left')
        self._freq_v = tk.StringVar(value=str(v.freq))
        fe = tk.Entry(mid, textvariable=self._freq_v,
                      bg='#21262d', fg=v.color, relief='flat',
                      font=('Consolas',10,'bold'), width=12,
                      insertbackground='white')
        fe.pack(side='left', padx=3)
        fe.bind("<Return>",   lambda _: self._apply_freq())
        fe.bind("<FocusOut>", lambda _: self._apply_freq())
        fe.bind("<Button-1>", lambda _: self._on_select(v.id))

        # Modo
        self._mode_v = tk.StringVar(value=v.mode)
        cb = ttk.Combobox(mid, textvariable=self._mode_v,
                          values=DEMOD_MODES, state='readonly', width=5)
        cb.pack(side='left', padx=2)
        cb.bind("<<ComboboxSelected>>", lambda _: self._apply_mode())

        bot = tk.Frame(self, bg='#161b22')
        bot.pack(fill='x', padx=4, pady=(0,3))

        # BW
        tk.Label(bot, text="BW:", bg='#161b22', fg='#6e7681',
                 font=('Consolas',8)).pack(side='left')
        self._bw_v = tk.StringVar(value=str(v.bw))
        bwe = tk.Entry(bot, textvariable=self._bw_v,
                       bg='#21262d', fg='#c9d1d9', relief='flat',
                       font=('Consolas',9), width=8,
                       insertbackground='white')
        bwe.pack(side='left', padx=3)
        bwe.bind("<Return>", lambda _: self._apply_bw())

        # Ganho AF
        tk.Label(bot, text="AF:", bg='#161b22', fg='#6e7681',
                 font=('Consolas',8)).pack(side='left', padx=(6,0))
        self._af_v = tk.DoubleVar(value=v.af_gain)
        tk.Scale(bot, from_=0.0, to=3.0, resolution=0.05,
                 orient='horizontal', variable=self._af_v,
                 length=70, bg='#161b22', fg='#8b949e',
                 highlightthickness=0, troughcolor='#21262d',
                 showvalue=False,
                 command=lambda _: self._apply_af()
                 ).pack(side='left')

    def mark_active(self, is_active: bool):
        col = self.vfo.color if is_active else '#30363d'
        self.config(highlightbackground=col)

    def _apply_freq(self):
        try:
            f = int(self._freq_v.get().replace(' ','').replace(',','').replace('.',''))
            self.vfo.freq = f
            self._on_change(self.vfo)
        except ValueError: pass

    def _apply_mode(self):
        self.vfo.mode = self._mode_v.get()
        self.vfo.rebuild()
        self._on_change(self.vfo)

    def _apply_bw(self):
        try:
            self.vfo.bw = max(2000, int(self._bw_v.get()))
            self.vfo.rebuild()
            self._on_change(self.vfo)
        except ValueError: pass

    def _apply_af(self):
        self.vfo.af_gain = self._af_v.get()

    def _toggle_active(self):
        self.vfo.active = self._active_v.get()

    def sync(self):
        """Atualiza campos a partir do estado interno do VFO."""
        self._freq_v.set(str(self.vfo.freq))
        self._mode_v.set(self.vfo.mode)
        self._bw_v.set(str(self.vfo.bw))
        self._af_v.set(self.vfo.af_gain)


# ─────────────────────────────────────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────────────────────────────────────
class SDRppBrownApp(tk.Tk):
    MAX_VFOS = 8

    def __init__(self):
        super().__init__()
        self.title("SDR++Brown Python  v3.0 — Multi-VFO")
        self.geometry("1400x880")
        self.configure(bg='#0d1117')
        self.minsize(1000,660)

        self.running     = False
        self.device      = SDRDevice()
        self.bookmarks   = BookmarkManager()
        self._vfos: list[VFO] = []
        self._vfo_panels: dict[int, VFOPanel] = {}
        self._active_id  = None
        self._center_hz  = 100_400_000
        self._fq         = queue.Queue(maxsize=4)
        self._aq         = queue.Queue(maxsize=8)
        self._worker     = None
        self._astream    = None
        self._spec       = None
        self._wfall      = None
        self._ruler      = None

        self._load_cfg()
        if not self._vfos:
            self._new_vfo(100_400_000, "FM", 180_000)

        self._style()
        self._build_ui()
        self._reload_bm()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    # ── VFO management ────────────────────────────────────────────────────────
    def _new_vfo(self, freq=None, mode="FM", bw=180_000):
        if len(self._vfos) >= self.MAX_VFOS:
            return None
        if freq is None:
            freq = self._center_hz
        v = VFO(freq, mode, bw)
        v.rebuild(self.device.rate, AUDIO_RATE)
        self._vfos.append(v)
        if self._active_id is None:
            self._active_id = v.id
        return v

    def _remove_vfo(self, vid):
        self._vfos = [v for v in self._vfos if v.id != vid]
        if vid in self._vfo_panels:
            self._vfo_panels[vid].destroy()
            del self._vfo_panels[vid]
        if self._active_id == vid:
            self._active_id = self._vfos[0].id if self._vfos else None
        if not self._vfos:
            self._new_vfo()
            self._refresh_vfo_list()
        self._refresh_active_marks()
        if self._spec: self._spec.redraw()

    def _set_active(self, vid):
        self._active_id = vid
        self._refresh_active_marks()
        if self._spec: self._spec.redraw()

    def _refresh_active_marks(self):
        for vid, panel in self._vfo_panels.items():
            panel.mark_active(vid == self._active_id)

    def _active_vfo(self) -> VFO | None:
        return next((v for v in self._vfos if v.id==self._active_id), None)

    def _set_vfo_freq(self, vid, hz):
        v = next((v for v in self._vfos if v.id==vid), None)
        if v:
            v.freq = max(1_000_000, min(2_000_000_000, int(hz)))
            if vid in self._vfo_panels:
                self._vfo_panels[vid].sync()
        if self._spec: self._spec.redraw()

    def _on_vfo_change(self, vfo: VFO):
        if self._spec: self._spec.redraw()

    # ── style ─────────────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self); s.theme_use('clam')
        BG='#0d1117'; SRF='#161b22'; T='#c9d1d9'; M='#8b949e'
        s.configure('.', background=BG, foreground=T, fieldbackground='#21262d')
        s.configure('TLabel',    background=BG,  foreground=T)
        s.configure('TFrame',    background=BG)
        s.configure('TButton',   background='#21262d', foreground=T,
                    borderwidth=1, relief='flat', padding=(8,4))
        s.map('TButton', background=[('active','#30363d')])
        s.configure('Go.TButton',   background='#1f6feb', foreground='white',
                    font=('Segoe UI',10,'bold'))
        s.map('Go.TButton',   background=[('active','#388bfd')])
        s.configure('Stop.TButton', background='#da3633', foreground='white',
                    font=('Segoe UI',10,'bold'))
        s.map('Stop.TButton', background=[('active','#f85149')])
        s.configure('TCombobox', fieldbackground='#21262d', foreground=T, arrowcolor=M)
        s.configure('TEntry',    fieldbackground='#21262d', foreground=T)
        s.configure('TNotebook', background=SRF)
        s.configure('TNotebook.Tab', background='#21262d', foreground=M, padding=[10,4])
        s.map('TNotebook.Tab',
              background=[('selected',SRF)], foreground=[('selected','#58a6ff')])
        s.configure('Treeview', background=BG, foreground=T,
                    fieldbackground=BG, rowheight=22, borderwidth=0)
        s.configure('Treeview.Heading', background=SRF, foreground=M)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Toolbar
        tb = tk.Frame(self, bg='#161b22', pady=4)
        tb.pack(side='top', fill='x')

        tk.Label(tb, text="📡 SDR++Brown", bg='#161b22', fg='#58a6ff',
                 font=('Segoe UI',13,'bold')).pack(side='left', padx=10)

        self._btn = ttk.Button(tb, text="▶  Iniciar", style='Go.TButton',
                               command=self._toggle)
        self._btn.pack(side='left', padx=6)

        self._status_lbl = tk.Label(tb, text="● PARADO", bg='#161b22',
                                    fg='#f85149', font=('Consolas',10,'bold'))
        self._status_lbl.pack(side='left', padx=8)

        self._hw_lbl = tk.Label(tb, text="[sem hardware]", bg='#161b22',
                                fg='#d29922', font=('Segoe UI',9))
        self._hw_lbl.pack(side='left', padx=4)

        for lbl,vn,fr,to,def_ in [("Min dB:","_vmin_v",-160,-20,-120),
                                    ("Max dB:","_vmax_v",-100,  0, -20)]:
            tk.Label(tb,text=lbl,bg='#161b22',fg='#8b949e').pack(side='left',padx=(10,2))
            v=tk.IntVar(value=def_); setattr(self,vn,v)
            tk.Scale(tb,from_=fr,to=to,orient='horizontal',variable=v,length=80,
                     bg='#161b22',fg='#8b949e',highlightthickness=0,
                     troughcolor='#21262d',command=self._range_changed).pack(side='left')

        # Toolbar: centro + SR
        tk.Label(tb,text="Centro (Hz):",bg='#161b22',fg='#8b949e').pack(side='left',padx=(12,2))
        self._center_v = tk.StringVar(value=str(self._center_hz))
        ce = tk.Entry(tb, textvariable=self._center_v,
                      bg='#21262d', fg='#c9d1d9', relief='flat',
                      font=('Consolas',10), width=12)
        ce.pack(side='left', padx=4)
        ce.bind("<Return>", lambda _: self._apply_center())

        tk.Label(tb,text="SR:",bg='#161b22',fg='#8b949e').pack(side='left',padx=(8,2))
        self._rate_v = tk.StringVar(value=str(self.device.rate))
        ttk.Combobox(tb, textvariable=self._rate_v,
                     values=["1024000","1800000","2048000","2400000","2880000","3200000"],
                     state='readonly', width=9).pack(side='left', padx=2)

        # Ganho
        tk.Label(tb,text="Ganho:",bg='#161b22',fg='#8b949e').pack(side='left',padx=(10,2))
        self._gain_v = tk.StringVar(value="Auto")
        ttk.Combobox(tb,textvariable=self._gain_v,values=GAIN_PRESETS,
                     state='readonly',width=6).pack(side='left',padx=2)

        ttk.Separator(self, orient='horizontal').pack(fill='x')

        # ── Pane principal ────────────────────────────────────────────────────
        pane = ttk.PanedWindow(self, orient='horizontal')
        pane.pack(fill='both', expand=True, padx=4, pady=4)

        # Coluna esquerda: visualização
        left = tk.Frame(pane, bg='#0d1117')
        pane.add(left, weight=5)

        self._spec = make_spectrum(
            left,
            lambda: self._vfos,
            lambda: self._active_id,
            self._set_active,
            self._set_vfo_freq,
            lambda: self._center_hz,
            lambda: self.device.rate,
        )
        self._spec.pack(fill='x', side='top')

        self._wfall = make_waterfall(left)
        self._wfall.pack(fill='both', expand=True, side='top')

        self._ruler = make_ruler(left,
                                 lambda: self._center_hz,
                                 lambda: self.device.rate)
        self._ruler.pack(fill='x', side='top')

        # Coluna direita: VFOs + tabs
        right = tk.Frame(pane, bg='#0d1117', width=280)
        pane.add(right, weight=1)

        # Cabeçalho VFOs
        vfohdr = tk.Frame(right, bg='#161b22', pady=3)
        vfohdr.pack(fill='x')
        tk.Label(vfohdr, text="VFOs", bg='#161b22', fg='#58a6ff',
                 font=('Segoe UI',10,'bold')).pack(side='left', padx=8)
        tk.Button(vfohdr, text="+ Novo VFO", bg='#1f6feb', fg='white',
                  relief='flat', font=('Segoe UI',9),
                  command=self._add_vfo_ui).pack(side='right', padx=6)

        # Scroll de VFOs
        vfo_outer = tk.Frame(right, bg='#0d1117')
        vfo_outer.pack(fill='x')
        vsc = tk.Scrollbar(vfo_outer, orient='vertical', bg='#21262d')
        vsc.pack(side='right', fill='y')
        self._vfo_canvas = tk.Canvas(vfo_outer, bg='#0d1117',
                                     highlightthickness=0, height=260,
                                     yscrollcommand=vsc.set)
        self._vfo_canvas.pack(side='left', fill='x', expand=True)
        vsc.config(command=self._vfo_canvas.yview)
        self._vfo_frame = tk.Frame(self._vfo_canvas, bg='#0d1117')
        self._vfo_canvas.create_window((0,0), window=self._vfo_frame,
                                       anchor='nw', tags='frame')
        self._vfo_frame.bind("<Configure>",
            lambda e: self._vfo_canvas.configure(
                scrollregion=self._vfo_canvas.bbox('all')))

        ttk.Separator(right, orient='horizontal').pack(fill='x')

        # Notebook abaixo dos VFOs
        nb = ttk.Notebook(right)
        nb.pack(fill='both', expand=True)
        tabs={}
        for nm in ("📌 Favoritos","⚙ SDR","ℹ Info"):
            f=ttk.Frame(nb); nb.add(f,text=nm); tabs[nm]=f
        self._build_bm_tab(tabs["📌 Favoritos"])
        self._build_sdr_tab(tabs["⚙ SDR"])
        self._build_info_tab(tabs["ℹ Info"])

        # Status bar
        sb = tk.Frame(self, bg='#161b22', height=20)
        sb.pack(side='bottom', fill='x')
        self._sb_freq = tk.Label(sb,bg='#161b22',fg='#8b949e',
                                  font=('Consolas',9),text="Centro: –")
        self._sb_freq.pack(side='left',padx=10)
        self._sb_vfo  = tk.Label(sb,bg='#161b22',fg='#58a6ff',
                                  font=('Consolas',9),text="")
        self._sb_vfo.pack(side='left',padx=10)
        tk.Label(sb,text="SDR++Brown v3 | Multi-VFO | RTL-SDR V3/V4",
                 bg='#161b22',fg='#21262d',font=('Segoe UI',8)
                 ).pack(side='right',padx=8)

        # Renderiza VFOs salvos
        self._refresh_vfo_list()

    # ── VFO list UI ───────────────────────────────────────────────────────────
    def _refresh_vfo_list(self):
        for w in list(self._vfo_panels.values()):
            w.destroy()
        self._vfo_panels.clear()
        for v in self._vfos:
            panel = VFOPanel(
                self._vfo_frame, v,
                on_select =self._set_active,
                on_remove =self._remove_vfo,
                on_change =self._on_vfo_change,
            )
            panel.pack(fill='x', padx=4, pady=2)
            self._vfo_panels[v.id] = panel
        self._refresh_active_marks()

    def _add_vfo_ui(self):
        av = self._active_vfo()
        freq = av.freq+200_000 if av else self._center_hz
        v = self._new_vfo(freq)
        if v is None:
            return   # MAX atingido
        panel = VFOPanel(
            self._vfo_frame, v,
            on_select=self._set_active,
            on_remove=self._remove_vfo,
            on_change=self._on_vfo_change,
        )
        panel.pack(fill='x', padx=4, pady=2)
        self._vfo_panels[v.id] = panel
        self._set_active(v.id)
        if self._spec: self._spec.redraw()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    def _build_bm_tab(self, p):
        ttk.Label(p,text="Favoritos",font=('Segoe UI',10,'bold')).pack(pady=(6,2))
        cols=("Nome","MHz","Modo")
        self._tree=ttk.Treeview(p,columns=cols,show='headings',height=8)
        for c in cols:
            self._tree.heading(c,text=c)
            self._tree.column(c,width=80)
        self._tree.pack(fill='both',expand=True,padx=4,pady=2)
        self._tree.bind("<Double-1>",lambda _:self._goto_bm())
        br=ttk.Frame(p); br.pack(fill='x',padx=4,pady=2)
        ttk.Button(br,text="+ Add",command=self._add_bm).pack(side='left',padx=2)
        ttk.Button(br,text="✕ Del",command=self._rm_bm ).pack(side='left',padx=2)
        ttk.Button(br,text="▶ Ir", command=self._goto_bm).pack(side='left',padx=2)
        ttk.Separator(p).pack(fill='x',padx=4,pady=4)
        ttk.Label(p,text="Presets:").pack(anchor='w',padx=4)
        pf=ttk.Frame(p); pf.pack(fill='x',padx=4)
        PRESETS=[("FM BR",100_900_000,"WFM",200_000),
                 ("Aviação",121_500_000,"AM",25_000),
                 ("ISS",145_800_000,"FM",15_000),
                 ("PMR446",446_006_250,"FM",12_500),
                 ("NOAA",137_620_000,"WFM",40_000),
                 ("ADS-B",1_090_000_000,"RAW",2_000_000)]
        for i,(nm,fr,mo,bw) in enumerate(PRESETS):
            ttk.Button(pf,text=nm,width=10,
                       command=lambda f=fr,m=mo,b=bw:self._quick(f,m,b)
                       ).grid(row=i//2,column=i%2,padx=2,pady=2,sticky='ew')

    def _build_sdr_tab(self, p):
        ttk.Label(p,text="RTL-SDR",font=('Segoe UI',10,'bold')).pack(pady=(6,4))
        r1=ttk.Frame(p); r1.pack(fill='x',padx=8,pady=2)
        ttk.Label(r1,text="PPM:").pack(side='left')
        self._ppm_v=tk.StringVar(value="0")
        pe=ttk.Entry(r1,textvariable=self._ppm_v,width=6); pe.pack(side='left',padx=4)
        pe.bind("<Return>",lambda _:self.device.set_ppm(int(self._ppm_v.get() or 0)))
        r2=ttk.Frame(p); r2.pack(fill='x',padx=8,pady=2)
        self._bias_v=tk.BooleanVar(value=False)
        ttk.Checkbutton(r2,text="Bias-Tee (V4)",variable=self._bias_v,
                        command=lambda:self.device.set_bias_tee(self._bias_v.get())
                        ).pack(side='left')
        ttk.Separator(p).pack(fill='x',padx=6,pady=6)
        self._hw_info=tk.Label(p,text=self.device.hw_label,bg='#0d1117',
                               fg='#3fb950',font=('Consolas',8),
                               wraplength=220,justify='left')
        self._hw_info.pack(anchor='w',padx=8)

    def _build_info_tab(self, p):
        tk.Label(p,text=(
            "\nSDR++Brown Python v3.0\n"
            "Multi-VFO Edition\n\n"
            "✓ Até 8 VFOs simultâneos\n"
            "✓ Cada VFO: modo/BW/AF/squelch\n"
            "✓ Arrastar VFO no espectro\n"
            "✓ Clicar seleciona VFO\n"
            "✓ BW sombreada por VFO\n"
            "✓ FM/WFM/AM/USB/LSB/CW/RAW\n"
            "✓ RTL-SDR V3 e V4\n"
            "✓ Bias-tee + PPM\n\n"
            "Licença: GPLv3"
        ), bg='#0d1117', fg='#8b949e',
           font=('Consolas',9), justify='left'
        ).pack(padx=10, pady=6, anchor='nw')

    # ── SDR loop ──────────────────────────────────────────────────────────────
    def _toggle(self):
        if not self.running: self._start()
        else:                self._stop()

    def _start(self):
        try: self.device.rate = int(self._rate_v.get())
        except: pass
        self.device.set_freq(self._center_hz)
        self.device.set_gain(self._gain_v.get())
        self.device.open()
        self._hw_lbl.config(text=self.device.hw_label,
                            fg='#3fb950' if not self.device.demo else '#d29922')
        try: self._hw_info.config(text=self.device.hw_label)
        except: pass
        for v in self._vfos:
            v.rebuild(self.device.rate, AUDIO_RATE)
        self.running = True
        self._btn.config(text="⏹  Parar", style='Stop.TButton')
        self._status_lbl.config(text="● RODANDO", fg='#3fb950')
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        if HAS_AUDIO: self._start_audio()

    def _stop(self):
        self.running = False
        if self._astream:
            try: self._astream.stop(); self._astream.close()
            except: pass
            self._astream = None
        self.device.close()
        self._btn.config(text="▶  Iniciar", style='Go.TButton')
        self._status_lbl.config(text="● PARADO", fg='#f85149')

    def _loop(self):
        while self.running:
            try:
                raw = self.device.read_samples(65536)
                # FFT global
                n   = min(FFT_SIZE, len(raw))
                win = np.blackman(n)
                sp  = np.fft.fftshift(np.fft.fft(raw[:n]*win, n=FFT_SIZE))
                fft_db = 20*np.log10(np.abs(sp)/n+1e-12)
                try: self._fq.put_nowait(fft_db.astype(np.float32))
                except queue.Full: pass

                # Mix áudio de VFOs ativos
                if HAS_AUDIO:
                    mix = None
                    for v in list(self._vfos):
                        if not v.active: continue
                        # shift IQ para centralizar VFO
                        offset = v.freq - self._center_hz
                        n2     = len(raw)
                        t      = np.arange(n2)/self.device.rate
                        shifted= raw * np.exp(-2j*np.pi*offset*t).astype(np.complex64)
                        audio  = v.process(shifted)
                        if mix is None: mix = audio
                        else:
                            ml = min(len(mix), len(audio))
                            mix = np.clip(mix[:ml]+audio[:ml], -1.0, 1.0)
                    if mix is not None and len(mix):
                        try: self._aq.put_nowait(mix)
                        except queue.Full: pass
            except Exception:
                time.sleep(0.05)

    def _start_audio(self):
        def cb(out,frames,_t,_s):
            try:
                chunk=self._aq.get_nowait()
                n=min(len(chunk),frames)
                out[:n,0]=chunk[:n]
                if n<frames: out[n:,0]=0.
            except queue.Empty: out.fill(0.)
        try:
            self._astream=sd.OutputStream(
                samplerate=AUDIO_RATE, channels=1,
                dtype='float32', blocksize=2048, callback=cb)
            self._astream.start()
        except Exception: pass

    # ── Tick ──────────────────────────────────────────────────────────────────
    def _tick(self):
        try:
            fft=self._fq.get_nowait()
            self._spec.update_fft(fft)
            self._wfall.push_fft(fft)
        except queue.Empty: pass
        if self._ruler: self._ruler.redraw()
        av = self._active_vfo()
        self._sb_freq.config(text=f"Centro: {_fmt_freq(self._center_hz)}")
        if av:
            self._sb_vfo.config(
                text=f"VFO{av.id}: {_fmt_freq(av.freq)}  {av.mode}  BW:{av.bw//1000}k")
        self.after(FRAME_MS, self._tick)

    # ── Controles gerais ──────────────────────────────────────────────────────
    def _apply_center(self):
        try:
            v = int(self._center_v.get().replace(' ',''))
            self._center_hz = v
            self.device.set_freq(v)
            if self._spec:  self._spec.redraw()
            if self._ruler: self._ruler.redraw()
        except ValueError: pass

    def _range_changed(self, _=None):
        mn=self._vmin_v.get(); mx=self._vmax_v.get()
        if self._spec:  self._spec.set_range(mn,mx)
        if self._wfall: self._wfall.set_range(mn,mx)

    # ── Bookmarks ─────────────────────────────────────────────────────────────
    def _reload_bm(self):
        self._tree.delete(*self._tree.get_children())
        for bm in self.bookmarks.items:
            self._tree.insert('','end',
                values=(bm['name'],f"{bm['freq']/1e6:.4f}",bm['mode']))

    def _add_bm(self):
        av=self._active_vfo()
        if av:
            self.bookmarks.add(f"BM_{av.freq//1000}k",av.freq,av.mode,av.bw)
            self._reload_bm(); self._save_cfg()

    def _rm_bm(self):
        sel=self._tree.selection()
        if sel:
            self.bookmarks.remove(self._tree.index(sel[0]))
            self._reload_bm(); self._save_cfg()

    def _goto_bm(self):
        sel=self._tree.selection()
        if sel:
            bm=self.bookmarks.items[self._tree.index(sel[0])]
            av=self._active_vfo()
            if av:
                av.freq=bm['freq']; av.mode=bm['mode']; av.bw=bm['bw']
                av.rebuild()
                if av.id in self._vfo_panels: self._vfo_panels[av.id].sync()
            if self._spec: self._spec.redraw()

    def _quick(self, freq, mode, bw):
        av=self._active_vfo()
        if av:
            av.freq=freq; av.mode=mode; av.bw=bw; av.rebuild()
            if av.id in self._vfo_panels: self._vfo_panels[av.id].sync()
            if self._spec: self._spec.redraw()

    # ── Config ────────────────────────────────────────────────────────────────
    def _save_cfg(self):
        try:
            with open(CONFIG_PATH,'w') as f:
                json.dump({
                    "center": self._center_hz,
                    "gain":   self._gain_v.get(),
                    "vfos":   [v.to_dict() for v in self._vfos],
                    "bookmarks": self.bookmarks.to_list(),
                }, f, indent=2)
        except Exception: pass

    def _load_cfg(self):
        try:
            with open(CONFIG_PATH) as f: c=json.load(f)
            self._center_hz = c.get("center", 100_400_000)
            for vd in c.get("vfos",[]):
                v=VFO.from_dict(vd)
                self._vfos.append(v)
                if self._active_id is None: self._active_id=v.id
            self.bookmarks.from_list(c.get("bookmarks",[]))
        except Exception: pass

    def _on_close(self):
        self._stop(); self._save_cfg(); self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _nice_step(x):
    if x<=0: return 1
    e=math.floor(math.log10(x)); b=10**e
    for m in(1,2,5,10):
        if b*m>=x: return b*m
    return b*10

def _fmt_freq(hz):
    hz=int(hz)
    if hz>=1_000_000_000: return f"{hz/1e9:.4f} GHz"
    if hz>=1_000_000:     return f"{hz/1e6:.4f} MHz"
    if hz>=1_000:         return f"{hz/1e3:.2f} kHz"
    return f"{hz} Hz"

FFT_SIZE = 4096

if __name__=="__main__":
    app=SDRppBrownApp()
    app.mainloop()
