import os
from dataclasses import dataclass, field
import uuid
from typing import List, Optional

cdir = os.path.dirname(os.path.realpath(__file__))


@dataclass
class WorldParamsConfig:
    R: int = 13
    T: float = 10.
    nb_channels: int = 1
    nb_dims: int = 2
    update_fn_version: str = 'v1'
    weighted_average: bool = True


@dataclass
class RunParamsConfig:
    code: str = str(uuid.uuid4())
    cells: str = ''
    max_run_iter: int = 1024
    nb_init_search: int = 128
    seed: int = 1


@dataclass
class RenderParamsConfig:
    pixel_border_size: int = 0
    pixel_size: int = 1
    pixel_size_power2: int = 2
    size_power2: int = 7
    world_size: Optional[List[int]] = field(default_factory=lambda: [128, 128])
    scale: float = 1.


@dataclass
class KernelParamsConfig:
    b: List[float] = field(default_factory=lambda: [1.0])
    c_in: int = 0
    c_out: int = 0
    gf_id: int = 0
    h: int = 1
    k_id: int = 0
    m: float = 0.3162686199839926
    q: float = 4.
    r: float = 1.
    s: float = 0.039730863904446516


@dataclass
class KernelsParamsConfig:
    k: List[KernelParamsConfig] = field(default_factory=lambda: [KernelParamsConfig()])


@dataclass
class LeniaxConfig:
    world_params: WorldParamsConfig = WorldParamsConfig()
    run_params: RunParamsConfig = RunParamsConfig()
    render_params: RenderParamsConfig = RenderParamsConfig()
    kernels_params: KernelsParamsConfig = KernelsParamsConfig()

    def __init__(self) -> None:
        print('woup')


@dataclass
class OtherConfig:
    dump_bests: bool = True


@dataclass
class LeniaxQDConfig(LeniaxConfig):
    other: OtherConfig = OtherConfig()
