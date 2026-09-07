"""Dependency smoke test for the selected RL framework."""


def test_torchrl_core_imports() -> None:
    import tensordict
    import torchrl
    from torchrl.data import LazyTensorStorage, ReplayBuffer

    assert torchrl.__version__
    assert tensordict.__version__
    assert ReplayBuffer is not None
    assert LazyTensorStorage is not None
