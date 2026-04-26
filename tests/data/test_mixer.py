import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.mixer import DataMixer, LossAdaptiveMixer, MixerConfig


def _make_loader(n: int, value: int, batch_size: int = 2) -> DataLoader:
    """Create a loader that yields tensors filled with `value`."""
    data = torch.full((n,), value, dtype=torch.float)
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_mixer_yields_batches():
    loaders = [_make_loader(10, 0), _make_loader(10, 1)]
    mixer = DataMixer(loaders, MixerConfig(weights=[0.5, 0.5]))
    batch = next(mixer)
    assert batch is not None


def test_mixer_weights_normalized():
    loaders = [_make_loader(10, 0), _make_loader(10, 1)]
    mixer = DataMixer(loaders, MixerConfig(weights=[2.0, 8.0]))
    probs = mixer.current_weights
    assert abs(sum(probs) - 1.0) < 1e-5
    assert abs(probs[0] - 0.2) < 0.01
    assert abs(probs[1] - 0.8) < 0.01


def test_mixer_samples_correct_distribution():
    """With extreme weights, almost all batches should come from source 0."""
    loaders = [_make_loader(100, 0), _make_loader(100, 1)]
    mixer = DataMixer(loaders, MixerConfig(weights=[100.0, 1.0]))
    counts = [0, 0]
    for i, batch in enumerate(mixer):
        if i >= 50:
            break
        val = batch[0][0].item()
        counts[int(val)] += 1
    assert counts[0] > counts[1]  # source 0 dominates


def test_mixer_resets_exhausted_loader():
    """Mixer should reset when a loader runs out."""
    loaders = [_make_loader(4, 0, batch_size=2), _make_loader(4, 1, batch_size=2)]
    mixer = DataMixer(loaders, MixerConfig(weights=[1.0, 0.0]))  # only source 0
    # Source 0 has 4 items -> 2 batches. Third should still work (reset).
    for i in range(6):  # 3x more than available batches
        batch = next(mixer)
        assert batch is not None


def test_mixer_update_weights():
    loaders = [_make_loader(10, 0), _make_loader(10, 1)]
    mixer = DataMixer(loaders, MixerConfig(weights=[0.5, 0.5]))
    mixer.update_weights([0.9, 0.1])
    assert abs(mixer.current_weights[0] - 0.9) < 0.01


def test_mixer_temperature_flattens():
    """High temperature should make weights more uniform."""
    loaders = [_make_loader(10, i) for i in range(3)]
    cfg_sharp = MixerConfig(weights=[10.0, 1.0, 1.0], temperature=0.1)
    cfg_flat = MixerConfig(weights=[10.0, 1.0, 1.0], temperature=10.0)
    mixer_sharp = DataMixer(loaders, cfg_sharp)
    mixer_flat = DataMixer(loaders, cfg_flat)
    # Flat mixer should have more uniform weights
    assert mixer_flat.current_weights[0] < mixer_sharp.current_weights[0]


def test_loss_adaptive_mixer_reweights():
    loaders = [_make_loader(10, 0), _make_loader(10, 1)]
    mixer = LossAdaptiveMixer(loaders, MixerConfig(weights=[0.5, 0.5]))
    # Record high loss for source 1
    for _ in range(20):
        mixer.record_loss(1, 10.0)
        mixer.record_loss(0, 0.1)
    # Source 1 should now have higher weight
    assert mixer.current_weights[1] > mixer.current_weights[0]
