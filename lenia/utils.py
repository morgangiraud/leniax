import os
# import datetime
import jax.numpy as jnp
from fractions import Fraction
from PIL import Image

cdir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(cdir, 'save')
image_ext = "png"

DIM_DELIM = {0: '', 1: '$', 2: '%', 3: '#', 4: '@A', 5: '@B', 6: '@C', 7: '@D', 8: '@E', 9: '@F'}


###
# Loader
###
def st2fracs2float(st):
    return [float(Fraction(st)) for st in st.split(',')]


def append_stack(list1, list2, count, is_repeat=False):
    list1.append(list2)
    if count != '':
        repeated = list2 if is_repeat else []
        list1.extend([repeated] * (int(count) - 1))


def recur_get_max_lens(dim, list1, max_lens, nb_dims):
    max_lens[dim] = max(max_lens[dim], len(list1))
    if dim < nb_dims - 1:
        for list2 in list1:
            recur_get_max_lens(dim + 1, list2, max_lens, nb_dims)


def recur_cubify(dim, list1, max_lens, nb_dims):
    more = max_lens[dim] - len(list1)
    if dim < nb_dims - 1:
        list1.extend([[]] * more)
        for list2 in list1:
            recur_cubify(dim + 1, list2, max_lens, nb_dims)
    else:
        list1.extend([0] * more)


def ch2val(c):
    if c in '.b':
        return 0
    elif c == 'o':
        return 255
    elif len(c) == 1:
        return ord(c) - ord('A') + 1
    else:
        return (ord(c[0]) - ord('p')) * 24 + (ord(c[1]) - ord('A') + 25)


def rle2arr(st, nb_dims, nb_channels):
    stacks = [[] for dim in range(nb_dims)]
    last, count = '', ''
    delims = list(DIM_DELIM.values())
    st = st.rstrip('!') + DIM_DELIM[nb_dims - 1]
    for ch in st:
        if ch.isdigit():
            count += ch
        elif ch in 'pqrstuvwxy@':
            last = ch
        else:
            if last + ch not in delims:
                append_stack(stacks[0], ch2val(last + ch) / 255, count, is_repeat=True)
            else:
                dim = delims.index(last + ch)
                for d in range(dim):
                    append_stack(stacks[d + 1], stacks[d], count, is_repeat=False)
                    stacks[d] = []
                # print('{0}[{1}] {2}'.format(last+ch, count, [np.asarray(s).shape for s in stacks]))
            last, count = '', ''

    cells = stacks[nb_dims - 1]
    max_lens = [0 for dim in range(nb_dims)]
    recur_get_max_lens(0, cells, max_lens, nb_dims)
    recur_cubify(0, cells, max_lens, nb_dims)

    cells = jnp.asarray(cells)
    cells = jnp.repeat(cells[jnp.newaxis, ...], nb_channels, axis=0)

    return cells


###
# Animals
###
def add_animal(cells, animal_cells, offset=None):
    # We ensure the animal and the world are compatible:
    # - same number of dims
    # - same number of channels
    assert len(cells.shape) == len(animal_cells.shape)
    assert cells.shape[0] == animal_cells.shape[0]

    if offset:
        assert len(cells.shape) == len(offset)
    else:
        offset = [0] * len(cells.shape)

    pads = []
    for i in range(len(cells.shape)):
        pad_start = (cells.shape[i] - animal_cells.shape[i]) // 2 - offset[i]
        pad_end = cells.shape[i] - animal_cells.shape[i] - pad_start

        pads.append((pad_start, pad_end))

    padded_animal_cells = jnp.pad(animal_cells, pads, mode='constant', constant_values=(0, 0))

    cells += padded_animal_cells

    return cells


###
# IMAGE
###
MARKER_COLORS_W = [0x5F, 0x5F, 0x5F, 0x7F, 0x7F, 0x7F, 0xFF, 0xFF, 0xFF]
MARKER_COLORS_B = [0x9F, 0x9F, 0x9F, 0x7F, 0x7F, 0x7F, 0x0F, 0x0F, 0x0F]


def save_image(save_dir, cells, vmin, vmax, pixel_size, pixel_border_size, colormap, animal_conf, idx, suffix=""):
    assert len(cells.shape) == 3, 'we only handle images under the format [C, H, W]'
    assert cells.shape[0] == 1, 'we only handle images with i channels'

    norm_cells = jnp.clip(normalize(cells[0], vmin, vmax), 0, 1)
    img = get_image(norm_cells, pixel_size, pixel_border_size, colormap)

    filename = f"{str(idx).zfill(3)}{suffix}.{image_ext}"
    # record_id = '{}-{}'.format(animal_conf['code'], datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))

    fullpath = os.path.join(save_dir, filename)

    img.save(fullpath)


def get_image(cells_buffer, pixel_size, pixel_border_size, colormap):
    y, x = cells_buffer.shape
    cells_buffer = jnp.repeat(cells_buffer, pixel_size, axis=0)
    cells_buffer = jnp.repeat(cells_buffer, pixel_size, axis=1)
    # zero = np.uint8(np.clip(normalize(0, vmin, vmax), 0, 1) * 252)
    zero = 0
    for i in range(pixel_border_size):
        cells_buffer[i::pixel_size, :] = zero
        cells_buffer[:, i::pixel_size] = zero

    img = colormap(cells_buffer)
    return Image.fromarray((img[:, :, :3] * 255).astype(jnp.uint8))


def normalize(v, vmin, vmax, is_square=False, vmin2=0, vmax2=0):
    if not is_square:
        return (v - vmin) / (vmax - vmin)
    else:
        return (v - vmin) / max(vmax - vmin, vmax2 - vmin2)


###
# Others
###
def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif not os.path.isdir(dir):
        raise Exception('The path provided ({}) exist and is not a dir, aborting'.format(dir))
