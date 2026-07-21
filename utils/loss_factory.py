import sys
sys.path.append("..")
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as Func


def multi_channel_stft(x, n_fft, hop_length, win_length, window, onesided=True):
    """x: (batch, channels, n_samples)"""
    bs, ch = x.shape[0], x.shape[1]
    x = x.view(bs*ch, -1)  # (B*C, L)

    X = torch.stft(x, n_fft, hop_length, win_length, window, onesided=onesided, return_complex=True)  # (B*C, F, T)
    X = X.view(bs, ch, X.shape[1], X.shape[2])  # (B,C,F,T), complex
    
    return X


class SISNRLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    
    def forward(self, y_pred, y_true):
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)

        loss =  - torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2 / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2+1e-8) + 1e-8).mean()
        
        return loss
    

class HybridLoss(nn.Module):
    def __init__(
        self,
        fft_len=0.032,
        hop_len=0.016,
        win_len=0.032,
        compress_factor=0.3,
        eps=1e-12,
        lamda_ri=30,
        lamda_mag=70,
        lamda_sdr=1,
    ):
        super().__init__()
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.win_len = win_len
        self.c = compress_factor
        self.eps = eps
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag
        self.lamda_sdr = lamda_sdr

    def forward(self, y_pred, y_true, fs):
        assert y_pred.shape[-1] == y_true.shape[-1]
        if y_pred.ndim == 3:
            y_pred = y_pred.squeeze(1)
        if y_true.ndim == 3:
            y_true = y_true.squeeze(1)
        
        device = y_true.device
        
        n_fft = int(self.fft_len * fs)
        n_hop = int(self.hop_len * fs)
        n_win = int(self.win_len * fs)
        pred_stft = torch.stft(y_pred, n_fft, n_hop, n_win, torch.hann_window(n_win).to(device), return_complex=True)
        true_stft = torch.stft(y_true, n_fft, n_hop, n_win, torch.hann_window(n_win).to(device), return_complex=True)

        pred_mag = torch.abs(pred_stft).clamp(self.eps)
        true_mag = torch.abs(true_stft).clamp(self.eps)
        
        pred_stft_c = pred_stft / pred_mag**(1 - self.c)
        true_stft_c = true_stft / true_mag**(1 - self.c)

        real_loss = torch.mean((pred_stft_c.real - true_stft_c.real)**2)
        imag_loss = torch.mean((pred_stft_c.imag - true_stft_c.imag)**2)
        mag_loss = torch.mean((pred_mag**self.c - true_mag**self.c)**2)

        # SISNR loss
        sdr = - 2*torch.log10(
            torch.norm(y_pred, dim=-1, keepdim=True) / 
            torch.norm(y_pred - y_true, dim=-1, keepdim=True).clamp(self.eps) + 
            self.eps
        ).mean()
        
        loss = self.lamda_ri*(real_loss + imag_loss) + self.lamda_mag*mag_loss + self.lamda_sdr*sdr
        
        return loss



class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_len=120, win_len=600, window="hann_window"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.register_buffer("window", getattr(torch, window)(win_len))

    def loss_spectral_convergence(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

    def loss_log_magnitude(self, x_mag, y_mag):
        return Func.l1_loss(torch.log(y_mag), torch.log(x_mag))

    def forward(self, x, y):
        """x, y: (B, T), in time domain"""
        assert x.shape == y.shape
        x = torch.stft(x, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        y = torch.stft(y, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        x_mag = torch.abs(x).clamp(1e-8)
        y_mag = torch.abs(y).clamp(1e-8)
        
        sc_loss = self.loss_spectral_convergence(x_mag, y_mag)
        mag_loss = self.loss_log_magnitude(x_mag, y_mag)
        # ri_loss = self.loss_real_imag(torch.view_as_real(x), torch.view_as_real(y))
        # print(sc_loss, mag_loss, 200*ri_loss)
        loss = sc_loss + mag_loss
        return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, hs, wl, window)]

    def forward(self, x, y):
        loss = 0.0
        for f in self.stft_losses:
            loss += f(x, y)
        loss /= len(self.stft_losses)
        return loss


class GesperLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
        n_bands=3
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        # self.fft_sizes = fft_sizes
        # self.hop_sizes = hop_sizes
        # self.win_lengths = win_lengths
        print('fft_sizes:', fft_sizes, 'hop_sizes:', hop_sizes, 'win_lengths:', win_lengths)
        # self.n_bands = n_bands
        # self.pqmf = PQMF(n_bands)
        # sub_fft_sizes = [item // n_bands for item in fft_sizes]
        # sub_hop_sizes = [item // n_bands for item in hop_sizes]
        # sub_win_lengths = [item // n_bands for item in win_lengths]
        self.multiR_stft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, window)
        # self.sub_multiR_stft_loss = MultiResolutionSTFTLoss(sub_fft_sizes, sub_hop_sizes, sub_win_lengths, window)

    def forward(self, x, y):
        loss = 0.0
        # xs = self.pqmf.analysis(x[:,None])
        # ys = self.pqmf.analysis(y[:,None])
        # for i in range(self.n_bands):
        #     loss += self.sub_multiR_stft_loss(xs[:,i], ys[:,i])
        # loss /= self.n_bands
        loss += self.multiR_stft_loss(x, y)
        return loss
    


if __name__=='__main__':
    a = torch.randn(2, 48000*2)
    b = torch.randn(2, 48000*2)
    
    loss_hyb = HybridLoss()
    loss = loss_hyb(a, b , 24000)
    print(loss)
    
    # loss_gesper = GesperLoss(sr=48000)
    # loss = loss_gesper(a, b)
    # print(loss)
