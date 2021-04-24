import torch
import pytest
import norbert


@pytest.fixture(params=[8, 11, 33])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[8, 11, 33])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def nb_sources(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def nb_iterations(request):
    return request.param


@pytest.fixture(params=[torch.complex64, torch.complex128])
def dtype(request):
    return request.param


@pytest.fixture
def X(request, nb_frames, nb_bins, nb_channels, dtype):
    Mix = torch.randn(nb_frames, nb_bins, nb_channels) + \
        torch.randn(nb_frames, nb_bins, nb_channels) * 1j
    return Mix.to(dtype)


@pytest.fixture
def V(request, nb_frames, nb_bins, nb_channels, nb_sources):
    return torch.rand(nb_frames, nb_bins, nb_channels, nb_sources)


def test_shapes(V, X):
    Y = norbert.residual_model(V, X)
    assert X.shape == Y.shape[:-1]

    Y = norbert.wiener(V, X)
    assert X.shape == Y.shape[:-1]

    Y = norbert.softmask(V, X)
    assert X.shape == Y.shape[:-1]


def test_wiener_copy(X, V):
    X0 = X.clone()
    V0 = V.clone()

    _ = norbert.wiener(V, X)

    assert torch.allclose(X0, X)
    assert torch.allclose(V0, V)


def test_softmask_copy(X, V):
    X0 = X.clone()
    V0 = V.clone()

    _ = norbert.softmask(V, X)

    assert torch.allclose(X0, X)
    assert torch.allclose(V0, V)


def test_residual_copy(X, V):
    X0 = X.clone()
    V0 = V.clone()

    _ = norbert.residual_model(V, X)

    assert torch.allclose(X0, X)
    assert torch.allclose(V0, V)


def test_silent_sources(X, V):
    V[..., :] = 0.0
    Y = norbert.softmask(V, X)

    assert X.shape == Y.shape[:-1]

    Y = norbert.wiener(V, X)
    assert X.shape == Y.shape[:-1]


def test_softmask(V, X):
    X = (X.shape[-1] * torch.ones(X.shape)).to(torch.complex128)
    Y = norbert.softmask(V, X)
    assert torch.allclose(Y.sum(-1), X)


def test_wiener(V, X):
    X = (X.shape[-1] * torch.ones(X.shape)).to(torch.complex128)
    Y = norbert.wiener(V, X)
    assert torch.allclose(Y.sum(-1), X)
