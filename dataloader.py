import os, csv
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

class AcousticScenesDatasetTA(Dataset):
    """
    Returns:
      features: FloatTensor (C, T, F) where channels are [logmel, Δ, ΔΔ] (or duplicated to N)
      label_idx: int
      meta: {"path": str, "label": str}
    """
    def __init__(
        self,
        root_dir: str,
        meta_filename: str = "meta.txt",
        audio_subdir: str = "audio",
        sr: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 882,    # ~20 ms @ 44.1 kHz
        win_length: int = 2048,   # ~46 ms
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        segment_seconds: Optional[float] = None,   # random crop length; None = full
        compute_deltas: bool = True,
        duplicate_channels_to: Optional[int] = None,  # e.g., 6
        seed: Optional[int] = None,
        warn_missing: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, meta_filename)
        self.audio_dir = os.path.join(root_dir, audio_subdir)
        self.sr = sr
        self.segment_seconds = segment_seconds
        self.compute_deltas = compute_deltas
        self.duplicate_channels_to = duplicate_channels_to
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Parse meta
        self.items: List[Tuple[str, str]] = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row: continue
                relpath = row[0].strip()
                label = row[1].strip() if len(row) > 1 else ""
                apath = relpath if os.path.isabs(relpath) else os.path.join(self.root_dir, relpath)
                if not os.path.exists(apath):
                    apath2 = os.path.join(self.audio_dir, os.path.basename(relpath))
                    if os.path.exists(apath2):
                        apath = apath2
                    elif warn_missing:
                        print(f"[WARN] missing audio: {relpath}")
                        continue
                self.items.append((apath, label))

        labels = sorted({lbl for _, lbl in self.items})
        self.label_to_index: Dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}
        self.index_to_label: List[str] = labels
        if not self.items:
            raise RuntimeError(f"No audio found via {self.meta_path}")

        # Transforms (work on mono waveform / feature tensors)
        self.resample = None  # lazily create only if needed per file
        self.melspec = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            f_min=fmin, f_max=fmax, n_mels=n_mels, power=2.0, center=True, norm=None, mel_scale="htk"
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)
        self.delta1 = T.ComputeDeltas(win_length=9) if compute_deltas else None
        self.delta2 = T.ComputeDeltas(win_length=9) if compute_deltas else None  # will apply to Δ

        self.hop_length = hop_length

    # ----- helpers -----

    def _load_mono(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)        # (C, N), float32
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)             # -> (1, N)

        # Downmix: average channels (robust across torchaudio versions)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # (1, N)

        # Resample if needed (use functional API; stateless & version-stable)
        if sr != self.sr:
            wav = AF.resample(wav, orig_freq=sr, new_freq=self.sr)  # (1, N')

        return wav.squeeze(0)  # -> (N,)

    def _features(self, wav: torch.Tensor) -> torch.Tensor:
        # mels: (n_mels, T)
        mel = self.melspec(wav)                # power mel
        mel_db = self.to_db(mel)               # log-mel in dB
        chans = [mel_db]

        if self.compute_deltas:
            d1 = self.delta1(mel_db)
            d2 = self.delta2(d1)
            chans += [d1, d2]

        X = torch.stack(chans, dim=0)          # (C=1|3, F, T)
        return X

    def _minmax01_per_channel(self, X: torch.Tensor) -> torch.Tensor:
        # X: (C, F, T)
        C = X.size(0)
        Xn = torch.empty_like(X, dtype=torch.float32)
        for c in range(C):
            x = X[c]
            mn = x.min()
            mx = x.max()
            if (mx - mn).abs() < 1e-8:
                Xn[c] = torch.zeros_like(x, dtype=torch.float32)
            else:
                Xn[c] = ((x - mn) / (mx - mn)).to(torch.float32)
        return Xn

    def _random_time_crop_frames(self, X: torch.Tensor, target_frames: int) -> torch.Tensor:
        # X: (C, F, T)
        C, F, T = X.shape
        if T <= target_frames:
            # right-pad with zeros
            pad = target_frames - T
            if pad > 0:
                pad_tensor = torch.zeros((C, F, pad), dtype=X.dtype)
                X = torch.cat([X, pad_tensor], dim=2)
            return X[:, :, :target_frames]
        start = torch.randint(low=0, high=T - target_frames + 1, size=(1,), generator=self.rng).item()
        return X[:, :, start:start + target_frames]

    # ----- dataset API -----

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        wav = self._load_mono(path)                 # (samples,)
        X = self._features(wav)                     # (C_base, F, T)
        X = self._minmax01_per_channel(X)           # (C_base, F, T)

        # Random crop to fixed frames if requested
        if self.segment_seconds is not None:
            target_frames = int(round(self.segment_seconds * self.sr / self.hop_length))
            X = self._random_time_crop_frames(X, target_frames)

        # (C, T, F)
        X = X.permute(0, 2, 1).contiguous()

        # Optionally duplicate channels up to N (e.g., N=6)
        if self.duplicate_channels_to is not None:
            base_c = X.size(0)
            if self.duplicate_channels_to % base_c != 0:
                raise ValueError(f"duplicate_channels_to ({self.duplicate_channels_to}) "
                                 f"must be a multiple of base channels ({base_c}).")
            reps = self.duplicate_channels_to // base_c
            X = X.repeat_interleave(reps, dim=0)    # (C*, T, F)

        y = self.label_to_index[label]
        meta = {"path": path, "label": label}
        return X, y, meta

    @property
    def class_names(self) -> List[str]:
        return self.index_to_label

    @property
    def num_classes(self) -> int:
        return len(self.index_to_label)
