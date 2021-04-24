import norbert
import musdb
import torch
import museval


def stft(x, n_fft=2048, n_hopsize=1024):
    return torch.stft(x, n_fft, n_hopsize, return_complex=True)


def istft(X, n_fft=2048, n_hopsize=1024):
    return torch.istft(X, n_fft, n_hopsize)


def oracle(track):
    # compute the mixture complex tf transform
    x = stft(torch.from_numpy(track.audio.T)).transpose(0, 2)
    v = []
    for name, value in track.sources.items():
        v_j = stft(torch.from_numpy(value.audio.T)).transpose(0, 2).abs() ** 2
        v += [v_j]
    v = torch.stack(v, 3)

    y = norbert.softmask(v, x).permute(3, 2, 1, 0)

    estimates = {}
    for j, (name, value) in enumerate(track.sources.items()):
        audio_hat = istft(y[j]).numpy().T
        estimates[name] = audio_hat

    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir=None
    )

    print(scores)

    return estimates


if __name__ == '__main__':
    mus = musdb.DB(download=True, subsets='test')

    for track in mus:
        oracle(track)
