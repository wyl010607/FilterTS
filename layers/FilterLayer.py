import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Complex_Func import complex_softmax, complex_relu

class FrequencyDomainFilterLayer(nn.Module):
    def __init__(self, seq_len, d_model, c_in, bandwidth=1, top_K_static_freqs=10, filter_type="all", quantile=0.90):
        super(FrequencyDomainFilterLayer, self).__init__()
        self.bandwidth = bandwidth
        self.top_K_static_freqs = top_K_static_freqs
        self.seq_len = seq_len
        self.d_model = d_model
        self.c_in = c_in
        self.filter_type = filter_type # "all" or "predefined" or "cross_variable"
        self.quantile = quantile
        if self.filter_type == "all":
            self.static_filter = PredefinedStaticFilter(seq_len, d_model, c_in, bandwidth, top_K_static_freqs)
            self.dynamic_cross_filter = DynamicCrossVariableFilter(d_model, c_in, quantile)
            self.mixing_factor = nn.Parameter(torch.zeros(c_in, 3), requires_grad=True)
        elif self.filter_type == "predefined":
            self.static_filter = PredefinedStaticFilter(seq_len, d_model, c_in, bandwidth, top_K_static_freqs)
            self.dynamic_cross_filter = None
            self.mixing_factor = nn.Parameter(torch.zeros(c_in, 2), requires_grad=True)
        elif self.filter_type == "cross_variable":
            self.static_filter = None
            self.dynamic_cross_filter = DynamicCrossVariableFilter(d_model, c_in, quantile)
            self.mixing_factor = nn.Parameter(torch.zeros(c_in, 2), requires_grad=True)
        else:
            raise ValueError("Invalid filter type")

    def forward(self, x):
        mixing_factor = torch.softmax(self.mixing_factor, dim=1)
        if self.filter_type == "all":
            static_filter = self.static_filter(x)
            dynamic_cross_filter = self.dynamic_cross_filter(x)
            x_out = x * mixing_factor[:, 0].unsqueeze(1)\
                    + static_filter * mixing_factor[:, 1].unsqueeze(1)\
                    + dynamic_cross_filter * mixing_factor[:, 2].unsqueeze(1)
        elif self.filter_type == "predefined":
            static_filter = self.static_filter(x)
            x_out = x * mixing_factor[:, 0].unsqueeze(1)\
                    + static_filter * mixing_factor[:, 1].unsqueeze(1)

        elif self.filter_type == "cross_variable":
            dynamic_cross_filter = self.dynamic_cross_filter(x)
            x_out = x * mixing_factor[:, 0].unsqueeze(1)\
                    + dynamic_cross_filter * mixing_factor[:, 1].unsqueeze(1)
        else:
            raise ValueError("Invalid filter type")
        return x_out


class PredefinedStaticFilter(nn.Module):
    def __init__(self, seq_len, d_model, c_in, bandwidth, top_K_static_freqs):
        super(PredefinedStaticFilter, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.c_in = c_in
        self.bandwidth = bandwidth
        self.top_K_static_freqs = top_K_static_freqs
        self.num_filters = top_K_static_freqs

        # Generate filters for the original sequence length
        self.filters = nn.Parameter(torch.zeros(c_in, d_model), requires_grad=False)  # # (c_in, d_model)

        # Learnable weights for combining the filters
        self.weights = nn.Parameter(torch.randn(c_in, d_model, dtype=torch.cfloat),
                                    requires_grad=True)
        self.amplitude_scalars = nn.Parameter(torch.ones(c_in, d_model), requires_grad=True)


    def forward(self, x):
        adjusted_signal = x * self.amplitude_scalars
        normed_weights = complex_softmax(complex_relu(self.weights), dim=1)
        #normed_weights = self.weights
        weighted_sum = self.filters * normed_weights
        filtered_signal = adjusted_signal * weighted_sum
        x_out = filtered_signal
        return x_out

    def generate_var_filters(self, static_freqs_idx):
        filters_list = []
        for i in static_freqs_idx:
            low_idx = int(max(0, i - self.bandwidth))
            #high_idx = min(self.seq_len//2 , i + self.bandwidth)
            high_idx = int(min(self.seq_len, i + self.bandwidth))
            filters_list.append(self.generate_bandpass_filter(low_idx, high_idx, self.seq_len))
        # Concatenate and resample the filters
        #filters = torch.cat(filters_list, dim=0)# (num_filters, seq_len)
        # sum of the filters for effectively calc (1, seq_len)
        filters = torch.sum(torch.stack(filters_list), dim=0)
        filters = self.resample_filter_fourier(filters, self.d_model) # (1, d_model)
        return filters


    def generate_bandpass_filter(self, low_idx, high_idx, seq_len):
        """Generate a band-pass filter between two cutoff frequencies."""
        filter_fft = torch.zeros(seq_len)
        filter_fft[low_idx:high_idx+1] = 1.0
        return filter_fft.unsqueeze(0)

    def set_static_freqs_idx(self, static_freqs_idx): # K * N
        self.static_freqs_idx = static_freqs_idx
        var_filters_list = []
        for i in range(self.c_in):
            var_filters = self.generate_var_filters(self.static_freqs_idx[:, i]) # (1, d_model)
            var_filters_list.append(var_filters)
        filters = torch.cat(var_filters_list, dim=0) # (c_in, d_model)
        self.filters = nn.Parameter(filters.to(self.filters.device), requires_grad=False)

    def resample_filter(self, filter, new_length):
        # Resample the filter similarly as the signal
        real_part = filter
        real_interpolated = F.interpolate(real_part.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
        return real_interpolated

    def resample_filter_fourier(self, filter, new_length):
        N, L = filter.shape
        if new_length > L:
            resampled_data = torch.zeros(N, new_length,
                                         device=filter.device)  # Prepare the new data array
            resampled_data[:, :L] = filter
        else:
            resampled_data = filter[:, :new_length]
        return resampled_data


    def fourier_interpolate(self, x_fft, new_length):
        B, N, L = x_fft.shape
        if new_length > L:
            # Upsampling: We keep all the original data and pad with zeros in high frequencies
            resampled_data = torch.zeros(B, N, new_length, dtype=torch.cfloat,
                                   device=x_fft.device)  # Prepare the new data array

            resampled_data[:, :, :L] = x_fft
        else:
            # Downsampling or keeping the length the same
            resampled_data = x_fft[:, :, :new_length]

        return resampled_data


class DynamicCrossVariableFilter(nn.Module):
    def __init__(self, d_model, c_in, quantile=0.90):
        super(DynamicCrossVariableFilter, self).__init__()
        self.amplitude_scalars = nn.Parameter(torch.ones(c_in, d_model), requires_grad=True)
        self.c_in = c_in
        self.threshold_quantile = quantile
        self.weights = nn.Parameter(torch.randn(c_in, c_in, dtype=torch.cfloat), requires_grad=True)
        self.mixing_factor = nn.Parameter(torch.zeros(c_in, 1, dtype=torch.cfloat) + 0.5, requires_grad=True)

    def forward(self, x):
        adjusted_signal = x * self.amplitude_scalars
        signal_magnitude = torch.abs(x)
        threshold = torch.quantile(signal_magnitude, self.threshold_quantile, dim=2, keepdim=True)
        above_threshold_mask = signal_magnitude > threshold
        masked_signal = x * above_threshold_mask
        normed_weights = complex_softmax(complex_relu(self.weights), dim=1)
        weighted_sum = torch.matmul(normed_weights, masked_signal.conj())
        filtered_signal = adjusted_signal * weighted_sum
        x_out = (1 - self.mixing_factor) * x + self.mixing_factor * filtered_signal
        return x_out
