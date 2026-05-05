import math
import os
import random
import json

import numpy as np
import torch
import torch.optim as optim

import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from diffusers import DDPMScheduler


def load_config(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_max_epoch(path):
    epoch = -1
    if not os.path.isdir(path):
        return epoch
    for name in os.listdir(path):
        if not name.endswith(".pkl"):
            continue
        stem = name[:-4]
        if stem.isdigit():
            epoch = max(epoch, int(stem))
    return epoch


def print_size(net):
    if net is None or not isinstance(net, torch.nn.Module):
        return
    module_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum(np.prod(p.size()) for p in module_parameters)
    print(f"{net.__class__.__name__} Parameters: {params / 1e6:.6f}M", flush=True)


def std_normal(size, device=None):
    """
    Generate the standard Gaussian variable of a certain size
    """

    if device is None:
        device = torch.device("cpu")
    return torch.normal(0, 1, size=size, device=device)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    assert diffusion_step_embed_dim_in % 2 == 0
    half_dim = diffusion_step_embed_dim_in // 2
    embed = math.log(10000.0) / (half_dim - 1)
    embed = torch.exp(
        torch.arange(half_dim, device=diffusion_steps.device, dtype=torch.float32)
        * -embed
    )
    embed = diffusion_steps.float() * embed
    return torch.cat((torch.sin(embed), torch.cos(embed)), dim=1)


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    beta = torch.linspace(beta_0, beta_T, T)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    beta_tilde = beta.clone()
    if T > 1:
        beta_tilde[1:] = beta[1:] * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:])

    sigma = beta_tilde
    return {
        "T": T,
        "Beta": beta,
        "Alpha": alpha,
        "Alpha_bar": alpha_bar,
        "Sigma": sigma,
    }




def get_optimizer(net, learning_rate, scheduler_type="cosine", T_max=100, eta_min=1e-6):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    else:
        scheduler = None
    return optimizer, scheduler


def get_mask_rm(sample, k):
    mask = torch.ones(sample.shape)
    length_index = torch.arange(mask.shape[0])
    for channel in range(mask.shape[1]):
        idx = torch.randperm(len(length_index))[:k]
        mask[:, channel][idx] = 0
    return mask


def get_mask_mnr(sample, k):
    mask = torch.ones(sample.shape)
    length_index = torch.arange(mask.shape[0])
    segments = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(segments)
        mask[:, channel][s_nan[0] : s_nan[-1] + 1] = 0
    return mask


def get_mask_bm(sample, k):
    mask = torch.ones(sample.shape)
    length_index = torch.arange(mask.shape[0])
    segments = torch.split(length_index, k)
    s_nan = random.choice(segments)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0] : s_nan[-1] + 1] = 0
    return mask


def get_mask_forecast(sample, k):
    mask = torch.ones(sample.shape)
    length_index = torch.arange(mask.shape[0])
    segments = torch.split(length_index, k)
    s_nan = segments[1]
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0] : s_nan[-1] + 1] = 0
    return mask


def get_mask(batch, masking, k):
    # train_caldas expects batch shape (B, T, C) and returns mask in (B, C, T).
    if masking == "rm":
        transposed_mask = get_mask_rm(batch[0], k)
    elif masking == "mnr":
        transposed_mask = get_mask_mnr(batch[0], k)
    elif masking == "bm":
        transposed_mask = get_mask_bm(batch[0], k)
    elif masking == "forecast":
        transposed_mask = get_mask_forecast(batch[0], k)
    else:
        raise ValueError(f"Unknown masking mode: {masking}")

    mask = transposed_mask.permute(1, 0)
    mask = mask.repeat(batch.size(0), 1, 1).float().to(batch.device)
    loss_mask = ~mask.bool()
    return mask, loss_mask


def frequency_finder(y, n_frequencies=None):
    batch_size, channels, length = y.shape

    omega = torch.fft.rfft(y)
    freqs = torch.fft.rfftfreq(length, d=1.0).to(y.device)
    mags = torch.abs(omega)

    diff_1 = torch.diff(torch.sign(torch.diff(mags, dim=2)), dim=2)
    peaks_mask = torch.sign(diff_1)[:, :, 1:] < 0
    adjusted_peaks_mask = torch.nn.functional.pad(peaks_mask, (2, 1), "constant", 0)
    peak_magnitudes = mags * adjusted_peaks_mask

    if n_frequencies is None:
        baseline = torch.mean(mags[:, :, 1:], dim=2) + torch.sqrt(torch.var(mags[:, :, 1:], dim=2))
        baseline = baseline.unsqueeze(2)
        n_frequencies = torch.sum(peak_magnitudes[:, :, 1:] > baseline, dim=2)
    else:
        n_frequencies = torch.tensor(n_frequencies, device=y.device).expand(batch_size, channels)

    max_freqs = int(max(1, n_frequencies.max().item()))
    h_p_m = torch.zeros(y.shape[0], y.shape[1], max_freqs, device=y.device)
    h_p_f = torch.zeros(y.shape[0], y.shape[1], max_freqs, device=y.device)
    h_p_c = torch.zeros(y.shape[0], y.shape[1], max_freqs, device=y.device, dtype=torch.complex64)
    h_p_i = torch.zeros(y.shape[0], y.shape[1], max_freqs, device=y.device, dtype=torch.long)

    for b in range(batch_size):
        for c in range(channels):
            count = int(n_frequencies[b, c].item())
            if count <= 0:
                continue
            top_mags, top_indices = peak_magnitudes[b, c].topk(count)
            h_p_m[b, c, :count] = top_mags
            h_p_f[b, c, :count] = freqs[top_indices]
            h_p_c[b, c, :count] = omega[b, c][top_indices]
            h_p_i[b, c, :count] = top_indices

    return h_p_f, h_p_m, h_p_i, h_p_c


def fixed_frequency_finder(y, n_frequencies):
    omega = torch.fft.rfft(y)
    freqs = torch.fft.rfftfreq(y.shape[2], d=1.0).to(y.device)
    mags = torch.abs(omega)

    top_k_values_mags, top_k_indices = mags[:, :, 1:].topk(n_frequencies)
    top_k_indices = top_k_indices + 1

    h_p_m = top_k_values_mags
    h_p_f = freqs[top_k_indices]
    h_p_i = top_k_indices.to(torch.int64)
    h_p_c = torch.gather(omega, dim=-1, index=top_k_indices)
    return h_p_f, h_p_m, h_p_i, h_p_c


def synthetic_fft(signal, n_frequencies=None, fixed=True):
    batch_size, channels, signal_length = signal.shape

    if n_frequencies is None:
        signal_freq, magnitude, h_p_i, h_p_c = frequency_finder(signal)
    else:
        if fixed:
            signal_freq, magnitude, h_p_i, h_p_c = fixed_frequency_finder(signal, n_frequencies-1)
        else:
            signal_freq, magnitude, h_p_i, h_p_c = frequency_finder(signal)

    _ = signal_freq
    magnitude = (magnitude / signal_length) * 2
    _ = magnitude

    fft_length = signal_length // 2 + 1
    top_k = h_p_i.shape[2]

    h_p_i_expanded = h_p_i.unsqueeze(-1)
    h_p_c_expanded = h_p_c.unsqueeze(-1)

    synth = torch.zeros(
        batch_size,
        channels,
        top_k,
        fft_length,
        dtype=torch.complex64,
        device=signal.device,
    )
    synth_complete = synth.scatter_(-1, h_p_i_expanded, h_p_c_expanded)

    if signal_length % 2 == 0:
        synth_y = torch.fft.irfft(synth_complete, dim=-1)
    else:
        synth_y = torch.fft.irfft(synth_complete, dim=-1, n=signal_length)

    output = torch.sum(synth_y, dim=2)
    remainder_component = (signal - output).unsqueeze(2)
    synth_y = torch.cat((synth_y, remainder_component), dim=2)
    return output, synth_y


def sort_components(components, components_var=None):
    if components_var is not None:
        _, sorted_indices = torch.sort(components_var, descending=False, dim=2)
    else:
        _, sorted_indices = torch.sort(components.var(dim=-1), descending=False, dim=2)
    sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, components.size(3))
    sorted_components = torch.gather(components, dim=2, index=sorted_indices_expanded)
    return sorted_components, sorted_indices


def masked_components_fft_amplitude(
    y,
    mask,
    max_components,
    masking="forecast",
    fixed=True,
    order="desc",
    decomposition_method="fft",
):
    y_clone = y.clone()
    sample_channels = y_clone.size(1)

    if decomposition_method != "fft":
        raise NotImplementedError("Only FFT decomposition is currently ported.")

    if masking == "forecast":
        input_for_cond_number = int(mask[0, 0].sum().item())
        masked_batch = y_clone[mask.bool()].reshape(y_clone.size(0), sample_channels, input_for_cond_number)
        if decomposition_method == 'wavelet':
                    _, components_masked = synthetic_wavelet(masked_batch,k=max_components,fixed=fixed)
        elif decomposition_method == 'fft':
            _, components_masked = synthetic_fft(masked_batch,max_components,fixed=fixed)
    else:
        if decomposition_method == 'wavelet':
            _, components_masked = synthetic_wavelet(y_clone*(1-mask).float(),k=max_components,fixed=fixed)
        elif decomposition_method == 'fft':
            _, components_masked = synthetic_fft(y_clone*(1-mask).float(),max_components,fixed=fixed)

    if order == "desc":
        sorted_components_masked, _ = sort_components(components_masked)
    else:
        sorted_components_masked, _ = sort_components(components_masked)

    dk = sorted_components_masked.var(dim=3)
    dk.clamp_min_(1e-5)
    dk = dk / dk.sum(dim=-1, keepdim=True)

    target_components = max_components + 1
    if dk.size(2) < target_components:
        dk_padded = torch.zeros(
            y.size(0),
            sample_channels,
            target_components,
            device=y.device,
            dtype=dk.dtype,
        )
        start = target_components - dk.size(2)
        dk_padded[:, :, start:] = dk
        return dk_padded
    if dk.size(2) > target_components:
        return dk[:, :, -target_components:]
    return dk


def adjust_component_steps2(components_steps, components):
    num_components = components.shape[2]
    batch = components.shape[0]
    components_variance = components.var(dim=3)
    non_zero_components = components_variance != 0
    max_non_zero = non_zero_components.sum(2).max(dim=1).values


    if num_components == 1:
        #number of components is 1, so we set all steps to 0 and variance to 1 to avoid issues with zero variance.
        components_steps.fill_(0)
        return components_steps, components_variance

    if non_zero_components.all():
        # all components are non-zero, so we can keep the original steps and variance.
        return components_steps, components_variance
    
    mask1 = components_steps >= num_components
    if mask1.any():
        components_steps[mask1] = torch.randint(
            low=0,
            high=num_components,
            size=components_steps[mask1].shape,
            device=components_steps.device,
        )

    lower_bounds = (num_components - max_non_zero).view(batch, 1, 1)
    mask2 = components_steps < lower_bounds
    if mask2.any():
        replacement = torch.zeros_like(components_steps[mask2])
        flat_lb = lower_bounds.expand_as(components_steps)[mask2]
        for i in range(replacement.shape[0]):
            lb = int(flat_lb[i].item())
            replacement[i] = torch.randint(low=lb-1, high=num_components, size=(1,), device=components_steps.device)
        components_steps[mask2] = replacement

    return components_steps.long(), components_variance


def forward_process_non_monte_carlo(y, components, diffusion_hyperparams, target_snr=1):
    batch_size, channels, _ = y.shape
    device = y.device

    dh = diffusion_hyperparams
    alpha_bar = dh["Alpha_bar"].to(device)
    t = dh["t"].view(batch_size).long().to(device)
    k = dh["k"].view(batch_size).long().to(device)

    gaussian_noise = torch.normal(0, 1, y.shape, device=device)
    sig_var = components.var(dim=-1)
    noise_var = sig_var / target_snr

    noise_var = noise_var.clamp_min(1e-5)
    noise_var[:,:,:] = noise_var[:,:,:]/ noise_var[:,:,:].sum(dim=-1,keepdim=True)
    b, c, n, l = components.shape
    k_expanded = k.view(b, 1, 1, 1).expand(b, c, 1, l)
    selected_components = torch.gather(components, 2, k_expanded)

    z_k_l = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1) * selected_components

    mask_mean = torch.zeros_like(components)
    mask_sigma = torch.zeros_like(noise_var)
    for sample in range(0, batch_size):
        components_length = torch.tensor(range(mask_mean.shape[2]))
        list_of_segments_index = torch.split(components_length, [k[sample]+1,mask_mean.shape[2]-k[sample]-1], dim=0)
        ones_tobe = list_of_segments_index[1]
        mask_mean[sample,:,ones_tobe] = 1
        mask_sigma[sample,:,list_of_segments_index[0]] = 1

    z_k_l = z_k_l.reshape(z_k_l.shape[0], z_k_l.shape[1], z_k_l.shape[3])
    z_k_l = (components * mask_mean).sum(dim=2) + z_k_l
    vb,vc,vcom = noise_var.shape

    variance_sigma_matrix = noise_var * mask_sigma
    dk_sum = variance_sigma_matrix.sum(dim=2).unsqueeze(-1)

    k_expanded_1 = k.view(b, 1, 1).expand(b, c, 1)
    last_non_zero_elements = torch.gather(variance_sigma_matrix, -1, k_expanded_1)

    scaled_variance = torch.sqrt(
        torch.clamp_min((-last_non_zero_elements * alpha_bar[t].view(-1, 1, 1)) + dk_sum, 1e-12)
    )

    epsilon = scaled_variance * gaussian_noise
    y_total_out = z_k_l + epsilon

    dh["components_var"] = noise_var
    return y_total_out, epsilon, scaled_variance, dh


def sweep_forward_process_non_monte_carlo(
    y_total,
    components_sorted,
    diffusion_hyperparams,
    target_snr=1,
    t_steps=50,
    k_steps=5,
    save_dir="debug_plot/forward_process_sweep",
    gif_name="y_total_out_first_sample.gif",
    fps=8,
):
    """
    Sweep dh['k'] then dh['t'] and call forward_process_non_monte_carlo at each pair.

    Ordering is lexicographic on (k, t), so (0, t_steps-1) is before (1, 0).
    """

    alpha_bar = diffusion_hyperparams["Alpha_bar"]
    if t_steps > alpha_bar.numel():
        raise ValueError(
            f"t_steps={t_steps} exceeds Alpha_bar length={alpha_bar.numel()}."
        )

    max_valid_k = components_sorted.size(2) - 1
    if (k_steps - 1) > max_valid_k:
        raise ValueError(
            f"k_steps={k_steps} requires max k={k_steps - 1}, but max valid k is {max_valid_k}."
        )

    os.makedirs(save_dir, exist_ok=True)

    batch_size = y_total.size(0)
    device = y_total.device
    first_sample_frames = []
    order_pairs = []

    for k_idx in range(k_steps):
        for t_idx in range(t_steps):
            dh_iter = dict(diffusion_hyperparams)
            dh_iter["t"] = torch.full(
                (batch_size, 1, 1), t_idx, dtype=torch.long, device=device
            )
            dh_iter["k"] = torch.full(
                (batch_size, 1, 1), k_idx, dtype=torch.long, device=device
            )

            y_total_out, _, _, _ = forward_process_non_monte_carlo(
                y_total,
                components_sorted,
                dh_iter,
                target_snr=target_snr,
            )

            output_path = os.path.join(save_dir, f"y_total_out_k{k_idx:02d}_t{t_idx:02d}.pt")
            torch.save(y_total_out.detach().cpu(), output_path)

            first_sample_frames.append(y_total_out[0].detach().cpu())
            order_pairs.append((k_idx, t_idx))

    torch.save(
        {
            "order_pairs": order_pairs,
            "t_steps": t_steps,
            "k_steps": k_steps,
        },
        os.path.join(save_dir, "sweep_metadata.pt"),
    )

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        first_frame = first_sample_frames[0]
        channels_to_plot = min(4, first_frame.shape[0])
        x_axis = torch.arange(first_frame.shape[-1]).numpy()

        fig, ax = plt.subplots(figsize=(9, 3.5))
        line_handles = []
        for ch in range(channels_to_plot):
            (line_handle,) = ax.plot(x_axis, first_frame[ch].numpy(), lw=1.2, label=f"ch {ch}")
            line_handles.append(line_handle)

        y_min = min(frame[:channels_to_plot].min().item() for frame in first_sample_frames)
        y_max = max(frame[:channels_to_plot].max().item() for frame in first_sample_frames)
        margin = 0.05 * max(1e-6, abs(y_max - y_min))
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.legend(loc="upper right")

        def _update(frame_idx):
            frame = first_sample_frames[frame_idx]
            for ch in range(channels_to_plot):
                line_handles[ch].set_ydata(frame[ch].numpy())
            k_idx, t_idx = order_pairs[frame_idx]
            ax.set_title(f"forward_process_non_monte_carlo | sample 0 | k={k_idx}, t={t_idx}")
            return line_handles

        animation = FuncAnimation(
            fig,
            _update,
            frames=len(first_sample_frames),
            interval=max(1, int(1000 / max(1, fps))),
            blit=False,
        )
        gif_path = os.path.join(save_dir, gif_name)
        animation.save(gif_path, writer="pillow", fps=fps)
        plt.close(fig)
    except Exception as exc:
        # Keep tensor outputs saved even if GIF export fails.
        print(f"GIF export failed: {exc}", flush=True)

    return {
        "save_dir": save_dir,
        "gif_path": os.path.join(save_dir, gif_name),
        "n_saved": len(order_pairs),
        "order_pairs": order_pairs,
    }


def apply_forward(
    signal,
    diffusion_hyperparams,
    diffusion_steps,
    components_steps,
    target_snr=1,
    max_components=None,
    monte_carlo=True,
    decomposition_method="fft",
):
    y_total = signal
    dh = diffusion_hyperparams
    fixed = dh.get("fixed", True)
    device = signal.device

    if max_components == 0:
        alpha_bar = dh["Alpha_bar"].to(device)
        dh["t"] = diffusion_steps
        dh["k"] = components_steps

        y_total_noise = torch.sqrt(1 - alpha_bar[diffusion_steps]) * std_normal(signal.shape, device=device)
        y_total_out = torch.sqrt(alpha_bar[diffusion_steps]) * signal + y_total_noise
        y_total_in = y_total_out
        ones = torch.ones(signal.shape[0], signal.shape[1], device=device)
        return y_total_out, y_total_in, y_total_noise, dh, ones

    if decomposition_method == "fft":
        _, components = synthetic_fft(y_total, max_components, fixed=fixed) # batch, channels, components, length
    elif decomposition_method == 'wavelet':
        _, components = synthetic_wavelet(y_total,k=max_components,fixed=fixed)
    else:
        raise NotImplementedError(f"Decomposition method {decomposition_method} is not implemented.")

    if monte_carlo:
        raise NotImplementedError("Monte Carlo forward process is not ported in this module.")

    components_steps, components_variance = adjust_component_steps2(components_steps, components)
    dh["t"] = diffusion_steps
    dh["k"] = components_steps
    dh["components_var"] = components_variance

    components_sorted, _ = sort_components(components, dh["components_var"])

    #target_n = max_components + 1
    #if components_sorted.size(2) < target_n:
    #    pad = torch.zeros(
    #        components_sorted.size(0),
    #        components_sorted.size(1),
    #        target_n - components_sorted.size(2),
    #        components_sorted.size(3),
    #        device=components_sorted.device,
    #        dtype=components_sorted.dtype,
    #    )
    #    components_sorted = torch.cat((components_sorted, pad), dim=2)
    #elif components_sorted.size(2) > target_n:
    #    components_sorted = components_sorted[:, :, -target_n:, :]

    y_total_out, y_total_noise, dk_scaled, dh = forward_process_non_monte_carlo(
        y_total,
        components_sorted,
        dh,
        target_snr,
    )
    y_total_in = y_total_out
    return y_total_out, y_total_in, y_total_noise, dh, dk_scaled


def weighted_mse_loss(epsilon_theta, transformed_x_noise, loss_mask=None, expanded_dk_scaled=1):
    if loss_mask is None:
        loss_mask = torch.ones_like(epsilon_theta, dtype=torch.bool)
    dk_scaled_expanded = expanded_dk_scaled.repeat(1, 1, loss_mask.shape[-1])[loss_mask]
    squared_errors = torch.pow(
        (epsilon_theta[loss_mask] - transformed_x_noise[loss_mask]) / dk_scaled_expanded,
        2,
    )
    return squared_errors.mean()


def training_loss_caldas(
    net,
    loss_fn,
    x,
    diffusion_hyperparams,
    max_components,
    only_generate_missing=1,
    monte_carlo=True,
    decomposition_method="fft",
):
    dh = diffusion_hyperparams
    t_total = dh["T"] 

    audio, cond, mask, loss_mask = x
    mask = None if only_generate_missing == 1 else mask
    b, _, _ = audio.shape
    device = audio.device

    diffusion_steps = torch.randint(t_total, size=(b, 1, 1), device=device)
    components_steps = torch.randint(max_components, size=(b, 1, 1), device=device)
    print(f"diffusion_steps: {diffusion_steps.view(-1).cpu().numpy()}")
    print(f"components_steps: {components_steps.view(-1).cpu().numpy()}")
    transformed_x, transformed_x_prior, transformed_x_noise, dh, dk_scaled = apply_forward(
        audio,
        dh,
        diffusion_steps,
        components_steps,
        max_components=max_components,
        monte_carlo=monte_carlo,
        decomposition_method=decomposition_method,
    )

    _ = transformed_x_prior

    #if only_generate_missing == 1:
    #    transformed_x = audio * mask.float() + transformed_x * (1 - mask).float()
    #    transformed_x_noise = audio * mask.float() + transformed_x_noise * (1 - mask).float()

    #diffusion_steps = dh["t"]
    #components_steps = dh["k"]
    epsilon_theta = net(
        (
            transformed_x,
            cond,
            mask,
            diffusion_steps.view(b, 1) + t_total * components_steps.view(b, 1),
        )
    )

    if isinstance(loss_fn, torch.nn.MSELoss):
        if loss_mask is not None:
            return loss_fn(epsilon_theta[loss_mask], transformed_x_noise[loss_mask])
        return loss_fn(epsilon_theta, transformed_x_noise)

    if only_generate_missing == 1:
        return weighted_mse_loss(
            epsilon_theta,
            transformed_x_noise,
            loss_mask=loss_mask,
            expanded_dk_scaled=dk_scaled,
        )
    return weighted_mse_loss(
        epsilon_theta,
        transformed_x_noise,
        loss_mask=None,
        expanded_dk_scaled=dk_scaled,
    )


def sampling_caldas2(
    net,
    size,
    diffusion_hyperparams,
    cond,
    mask,
    max_components,
    dk=None,
    only_generate_missing=1,
    guidance_weight=1,
    max_components_gen=7,
    sampling_with_dk=1,
    dynamic=True,
):
    _ = dynamic

    if sampling_with_dk == 1:
        dk = torch.ones_like(cond[:, :, :max_components_gen]) / max(1, max_components)
    elif isinstance(sampling_with_dk, list):
        dk_list = torch.tensor(sampling_with_dk, dtype=cond.dtype, device=cond.device)
        dk = dk_list.view(1, 1, -1).repeat(cond.size(0), cond.size(1), 1)

    if dk is None:
        dk = torch.ones_like(cond[:, :, :max_components_gen])

    dh = diffusion_hyperparams
    t_total = dh["T"]
    alpha = dh["Alpha"].to(cond.device)
    alpha_bar = dh["Alpha_bar"].to(cond.device)
    sigma = dh["Sigma"].to(cond.device)

    assert len(size) == 3

    steps = []
    b, c, tt = cond.size()
    n = dk.size(2)

    x_all = torch.empty(b, c, n, tt, device=dk.device)
    for idx in range(n):
        std = torch.sqrt(dk[:, :, idx]).unsqueeze(-1)
        x_all[:, :, idx, :] = std * std_normal(cond.size(), device=cond.device)

    x = x_all.sum(dim=2)
    cicle_x = std_normal(cond.size(), device=cond.device)
    old_x = torch.zeros_like(x)

    with torch.no_grad():
        for k in range(max_components, max_components - max_components_gen, -1):
            for t in range(t_total - 1, -1, -1):
                if only_generate_missing == 1:
                    if guidance_weight == 0:
                        x = x * (1 - mask).float()
                    elif guidance_weight == 1:
                        x = x * (1 - mask).float() + cond * mask.float()
                    else:
                        raise ValueError("guidance_weight should be 0 or 1")

                steps.append(x.clone().detach().cpu().numpy())
                diffusion_steps = (t + k * t_total) * torch.ones((size[0], 1), device=cond.device)
                epsilon_theta = net((x, cond, mask, diffusion_steps))

                if k == max_components:
                    cicle_x = (cicle_x - ((1 - alpha[t]) / (1 - alpha_bar[t])) * epsilon_theta) / torch.sqrt(alpha[t])
                    x = cicle_x.clone().detach()
                elif k > 0:
                    cicle_x = (cicle_x - ((1 - alpha[t]) / (1 - alpha_bar[t])) * epsilon_theta) / torch.sqrt(alpha[t])
                    x = old_x.clone().detach() + cicle_x.clone().detach()
                else:
                    x = x - ((1 - alpha[t]) / (1 - alpha_bar[t])) * epsilon_theta

                if (t > 0) and (k > -1):
                    x = (
                        x
                        + torch.sqrt((dk[:, :, k].unsqueeze(-1)) * sigma[t]) * std_normal(size, device=cond.device)
                        + torch.sqrt(dk[:, :, :k].sum(dim=-1)).unsqueeze(-1)
                        * std_normal(size, device=cond.device)
                    )
                elif (t == 0) and (k > (max_components - max_components_gen) + 1):
                    old_x = x.clone().detach()
                    cicle_x = torch.zeros_like(x)

    return x, steps




def synthetic_wavelet(signal, k,fixed=True, J=8, wave='db4'):
    """
    Parallel to synthetic_fft, but using wavelets (pytorch_wavelets).
    
    Assumes wavelet_finder / fixed_wavelet_finder return:
       coeff_locs: (batch, channels, K, 2)   -> (scale, position)
       coeff_vals: (batch, channels, K)
    """
    batch, channels, length = signal.shape
    
    dwt = DWT1DForward(J=J, wave=wave).to(signal.device)
    idwt = DWT1DInverse(wave=wave).to(signal.device)
    
    if not fixed:
        raise NotImplementedError("Only fixed wavelet finder is implemented.")
    
    # ---------------------------
    # 2) Compute full DWT template
    # ---------------------------
    coeffs = dwt(signal)

    #create a single list of all coefficients
    coeffc_l = []
    coeffc_l.append(coeffs[0])
    for j in range(len(coeffs[1])):
        coeffc_l.append(coeffs[1][j])

    # clone empty coefficients
    empty_yl = torch.zeros_like(coeffs[0])
    empty_yh = [torch.zeros_like(h) for h in coeffs[1]]

    
    variances = torch.stack([torch.var(coeffc_l[i], dim=-1) for i in range(len(coeffc_l))]) # components, length, channels

    # Sort component indices by variance (descending)
    #idx_sorted = torch.argsort(variances, descending=True)
    top_k_indices = torch.topk(variances, k,dim=0).indices
    # this 4 here is hardcoded, but you can increase it to match k. 
    # Select top-k
    #keep_idx = idx_sorted[:k]
    # ---------------------------
    # 3) Reconstruct each component separately
    # ---------------------------
    if length % 2 != 0:
        length += 1  # adjust length for odd case
    components = torch.zeros(batch, channels, k, length, device=signal.device)

    for kk in range(k):
        # zero all coefficients
        yl = empty_yl.clone()
        yh = [h.clone() for h in empty_yh]

        # fill exactly ONE coefficient
        for b in range(batch):
            for c in range(channels):
                pos = top_k_indices[kk,b,c]
                if pos == 0:
                    yl[b, c, :] = coeffc_l[pos][b, c]
                else:
                    yh[pos-1][b, c] = coeffc_l[pos][b, c]

        # inverse wavelet
        recon = idwt((yl, yh))

        components[:, :, kk, :] = recon

    # ---------------------------
    # 4) Sum components
    # ---------------------------
    output = components.sum(dim=2)
    if signal.shape[-1] % 2 != 0:
        output = output[..., :-1]
        components = components[..., :-1]
    # ---------------------------
    # 5) Add remainder
    # ---------------------------
    remainder = (signal - output).unsqueeze(2)

    components = torch.cat((components, remainder), dim=2)

    return output, components



def calc_diffusion_hyperparams2(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    scheduler = DDPMScheduler(
        num_train_timesteps=T,
        beta_start=beta_0,
        beta_end=beta_T,
        beta_schedule="linear"
    )

    Alpha = scheduler.alphas
    Alpha_bar = scheduler.alphas_cumprod
    Beta = scheduler.betas

    Sigma = (((1-scheduler.alphas_cumprod[1:])*scheduler.betas[0:T-1]) / (1-scheduler.alphas_cumprod[0:T-1]))
    Sigma = torch.cat((torch.zeros(1), Sigma))

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    #diffusion_hyperparams = _dh
    return _dh
