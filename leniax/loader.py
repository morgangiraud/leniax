# import time
import io
import binascii
import itertools
import os
import codecs
import pickle
import gzip
import jax.numpy as jnp
import numpy as np
from typing import List, Any, Dict

from .constant import NB_CHARS

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')
image_ext = "png"


###
# Loader
###
def append_stack(list1: List, elem: Any, count, is_repeat=False):
    list1.append(elem)
    if count != '':
        repeated = elem if is_repeat else []
        list1.extend([repeated] * (int(count) - 1))


def recur_get_max_lens(dim, list1, max_lens, nb_dims: int):
    max_lens[dim] = max(max_lens[dim], len(list1))
    if dim < nb_dims - 1:
        for list2 in list1:
            recur_get_max_lens(dim + 1, list2, max_lens, nb_dims)


def recur_cubify(dim, list1, max_lens, nb_dims: int):
    more = max_lens[dim] - len(list1)
    if dim < nb_dims - 1:
        list1.extend([[]] * more)
        for list2 in list1:
            recur_cubify(dim + 1, list2, max_lens, nb_dims)
    else:
        list1.extend([0] * more)


def _recur_drill_list(dim, lists, row_func, nb_dims: int):
    if dim < nb_dims - 1:
        return [_recur_drill_list(dim + 1, e, row_func, nb_dims) for e in lists]
    else:
        return row_func(lists)


def _recur_join_st(dim, lists, row_func, nb_dims: int):
    if dim < nb_dims - 1:
        return DIM_DELIM[nb_dims - 1 - dim].join(_recur_join_st(dim + 1, e, row_func, nb_dims) for e in lists)
    else:
        return DIM_DELIM[nb_dims - 1 - dim].join(row_func(lists))


def make_array_compressible(cells: jnp.ndarray) -> jnp.ndarray:
    max_val = NB_CHARS**2 - 1

    cells_int32 = jnp.array(jnp.round(cells * max_val), dtype=jnp.int32)
    compressible_cells = jnp.array(cells_int32 / max_val, dtype=jnp.float32)

    return compressible_cells


def compress_array(cells: jnp.ndarray) -> str:
    max_val = NB_CHARS**2 - 1
    cells_int32 = cells_int32 = jnp.array(jnp.round(cells * max_val), dtype=jnp.int32)

    cells_shape = cells_int32.shape
    cells_shape_bytes = bytes(0)
    for i in cells_shape:
        cells_shape_bytes += int.to_bytes(i, length=4, byteorder='little')

    cells_int32_flatten = cells_int32.flatten()
    nb_int32_bytes = int.to_bytes(len(cells_int32_flatten), length=4, byteorder='little')

    gzip_bytes = gzip.compress(nb_int32_bytes + cells_int32_flatten.tobytes() + cells_shape_bytes)
    b64_str = str(codecs.encode(gzip_bytes, 'base64'), 'utf-8')

    return b64_str


def decompress_array(string_cells: str, nb_dims: int = 0) -> jnp.ndarray:
    try:
        cells = decompress_array_gzip(string_cells)
    except Exception:
        try:
            string_array = string_cells.split('::')
            if len(string_array) != 2 and len(string_array[0]) % 2 == 0:
                raise Exception()
            max_val = NB_CHARS**2 - 1
            cells_shape = [int(c) for c in string_array[-1].split(";")]
            cells_val_l = [ch2val(string_array[0][i:i + 2]) for i in range(0, len(string_array[0]), 2)]
            cells_int32 = jnp.array(cells_val_l, dtype=jnp.int32).reshape(cells_shape)
            cells = jnp.array(cells_int32 / max_val, dtype=jnp.float32)
        except Exception:
            try:
                string_bytes = io.BytesIO(string_cells.encode('latin1'))
                cells_uint8 = np.load(string_bytes)['x']  # type: ignore
                cells = cells_uint8 / 255.
            except Exception:
                try:
                    cells = decompress_array_base64(string_cells)
                except binascii.Error:
                    cells = deprecated_decompress_array(string_cells, nb_dims)

    return jnp.array(cells, dtype=jnp.float32)


def decompress_array_gzip(b64_str: str) -> jnp.ndarray:
    gzip_bytes = codecs.decode(bytes(b64_str, 'utf-8'), 'base64')

    cells_int32_bytes = gzip.decompress(gzip_bytes)
    max_val = NB_CHARS**2 - 1

    int_list = []
    nb_int32 = int.from_bytes(cells_int32_bytes[0:4], 'little')
    for i in range(1 * 4, (1 + nb_int32) * 4, 4):
        int_list.append(int.from_bytes(cells_int32_bytes[i:i + 4], 'little'))
    shape = []
    for i in range((1 + nb_int32) * 4, len(cells_int32_bytes), 4):
        shape.append(int.from_bytes(cells_int32_bytes[i:i + 4], 'little'))
    cells_int32 = jnp.array(int_list, dtype=jnp.int32).reshape(shape)
    cells = jnp.array(cells_int32 / max_val, dtype=jnp.float32)

    return cells


def decompress_array_base64(string_cells: str) -> jnp.ndarray:
    serialized_cells = codecs.decode(string_cells.encode(), "base64")
    cells = pickle.loads(serialized_cells)

    return jnp.array(cells, dtype=jnp.float32)


def ch2val(c: str) -> int:
    assert len(c) == 2
    first_char = c[0]
    second_char = c[1]

    if ord(first_char) >= ord('À'):
        first_char_idx = ord(first_char) - ord('À') + (ord('Z') - ord('A')) + (ord('z') - ord('a'))
    elif ord(first_char) >= ord('a'):
        first_char_idx = ord(first_char) - ord('a') + (ord('Z') - ord('A'))
    else:
        first_char_idx = ord(first_char) - ord('A')

    if ord(second_char) >= ord('À'):
        second_char_idx = ord(second_char) - ord('À') + (ord('Z') - ord('A')) + (ord('z') - ord('a'))
    elif ord(second_char) >= ord('a'):
        second_char_idx = ord(second_char) - ord('a') + (ord('Z') - ord('A'))
    else:
        second_char_idx = ord(second_char) - ord('A')

    return first_char_idx * NB_CHARS + second_char_idx


def val2ch(v: int) -> str:
    first_char_idx = v // NB_CHARS
    second_char_idx = v % NB_CHARS
    # We do this trick to avoid the special characters between
    if ord('A') + first_char_idx <= ord('Z'):
        first_char = chr(ord('A') + first_char_idx)
    elif ord('a') + first_char_idx - (ord('Z') - ord('A')) <= ord('z'):
        first_char = chr(ord('a') + first_char_idx - (ord('Z') - ord('A')))
    else:
        first_char = chr(ord('À') + first_char_idx - (ord('Z') - ord('A')) - (ord('z') - ord('a')))

    if ord('A') + second_char_idx <= ord('Z'):
        second_char = chr(ord('A') + second_char_idx)
    elif ord('a') + second_char_idx - (ord('Z') - ord('A')) <= ord('z'):
        second_char = chr(ord('a') + second_char_idx - (ord('Z') - ord('A')))
    else:
        second_char = chr(ord('À') + second_char_idx - (ord('Z') - ord('A')) - (ord('z') - ord('a')))

    return first_char + second_char


###
# Deprecated, kept for backward compatibilities
###
DIM_DELIM = {0: '', 1: '$', 2: '%', 3: '#', 4: '@A', 5: '@B', 6: '@C', 7: '@D', 8: '@E', 9: '@F'}


def char2val_deprecated(ch: str) -> int:
    if ch in '.b':
        return 0
    elif ch == 'o':
        return 255
    elif len(ch) == 1:
        return ord(ch) - ord('A') + 1
    else:
        return (ord(ch[0]) - ord('p')) * 24 + (ord(ch[1]) - ord('A') + 25)


def val2char_deprecated(v: int) -> str:
    if v == 0:
        return '.'
    elif v < 25:
        return chr(ord('A') + v - 1)
    else:
        return chr(ord('p') + (v - 25) // 24) + chr(ord('A') + (v - 25) % 24)


def deprecated_compress_array(cells):
    def drill_row(row):
        return [(len(list(g)), val2char_deprecated(v).strip()) for v, g in itertools.groupby(row)]

    def join_row_shorten(row):
        return [(str(n) if n > 1 else '') + c for n, c in row]

    nb_dims = len(cells.shape)
    cells_int_l = np.rint(cells * 255).astype(int).tolist()  # [[255 255] [255 0]]

    rle_groups = _recur_drill_list(0, cells_int_l, drill_row, nb_dims)
    st = _recur_join_st(0, rle_groups, join_row_shorten, nb_dims)  # "2 yO $ 1 yO"

    return st + '!'


def deprecated_decompress_array(cells_code: str, nb_dims: int) -> jnp.ndarray:
    stacks: List[List] = [[] for dim in range(nb_dims)]
    last, count = '', ''
    delims = list(DIM_DELIM.values())
    st = cells_code.rstrip('!') + DIM_DELIM[nb_dims - 1]
    for ch in st:
        if ch.isdigit():
            count += ch
        elif ch in 'pqrstuvwxy@':
            last = ch
        else:
            if last + ch not in delims:
                append_stack(stacks[0], char2val_deprecated(last + ch) / 255, count, is_repeat=True)
            else:
                dim = delims.index(last + ch)
                for d in range(dim):
                    append_stack(stacks[d + 1], stacks[d], count, is_repeat=False)
                    stacks[d] = []
                # print('{0}[{1}] {2}'.format(last+ch, count, [np.asarray(s).shape for s in stacks]))
            last, count = '', ''

    cells_l = stacks[nb_dims - 1]
    max_lens = [0 for dim in range(nb_dims)]
    recur_get_max_lens(0, cells_l, max_lens, nb_dims)
    recur_cubify(0, cells_l, max_lens, nb_dims)

    cells = jnp.array(cells_l, dtype=jnp.float32)

    return cells


def load_raw_cells(config: Dict, use_init_cells: bool = True) -> jnp.ndarray:
    nb_dims = config['world_params']['nb_dims']
    if use_init_cells is True:
        if 'init_cells' not in config['run_params']:
            # Backward compatibility
            cells = config['run_params']['cells']
        else:
            cells = config['run_params']['init_cells']
    else:
        cells = config['run_params']['cells']
    if type(cells) is str:
        if cells == 'MISSING':
            cells = jnp.array([])
        elif cells == 'last_frame.p':
            with open(os.path.join(config['main_path'], 'last_frame.p'), 'rb') as f:
                cells = jnp.array(pickle.load(f), dtype=jnp.float32)
        else:
            cells = decompress_array(cells, nb_dims + 1)  # we add the channel dim
    elif type(cells) is list:
        cells = jnp.array(cells, dtype=jnp.float32)

    # We repair the missing channel in case of the single channel got squeezed out
    if len(cells.shape) == nb_dims and config['world_params']['nb_channels'] == 1:
        cells = jnp.expand_dims(cells, 0)

    return cells
