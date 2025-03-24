import torch


def get_topK_static_freqs_idx(time_series, topK, seq_len):
    # to tensor
    time_series = torch.tensor(time_series, dtype=torch.float32)
    # time_series: (L, N)
    L, N = time_series.shape
    x_fft = torch.fft.rfft(time_series, dim=0)
    # amplitude
    x_amp = torch.abs(x_fft)
    # window = L // seq_len
    window = L // (seq_len * 2)
    window_num = x_amp.shape[0] // window
    # get the max amplitude in each window use max not mean
    x_amp = x_amp[:window * window_num].reshape(window, window_num, N).max(dim=0).values
    #x_amp[:window * window_num].reshape(window, window_num, N).mean(dim=0).values
    # get topK index
    topK_idx = torch.topk(x_amp, topK, dim=0).indices
    # return tensor(topK, N)
    return topK_idx
