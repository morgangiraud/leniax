import os
import time
import ffmpeg
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Union, Any

from .utils import get_image


def dump_video(
    save_dir: str,
    all_cells: jnp.ndarray, 
    render_params: Dict, 
    colormaps: Union[List, Any], 
    prefix: str = '',
    transparent_bg: bool = False
):
    assert len(all_cells.shape) == 4  # [nb_iter, C, H, W]
    if type(colormaps) != list:
        colormaps = [colormaps]
    if prefix == '':
        prefix = 'beast'

    np_all_cells = np.array(all_cells)
    nb_iter_done = len(np_all_cells)
    width = np_all_cells[0].shape[-1] * render_params['pixel_size']
    height = np_all_cells[0].shape[-2] * render_params['pixel_size']

    all_outputs_fullpath = []
    for colormap in colormaps:
        process = ffmpeg.input(
            'pipe:',
            format='rawvideo',
            pix_fmt='rgba',
            s=f"{width}x{height}",
            framerate=30,
        )

        if transparent_bg:
            output_fullpath = os.path.join(save_dir, f"{prefix}_{colormap.name}_{width}_{height}.mkv")  # type: ignore
            process = process.output(
                output_fullpath, vcodec="ffv1"
            ).overwrite_output().run_async(
                pipe_stdin=True, quiet=True
            )

        else:
            output_fullpath = os.path.join(save_dir, f"{prefix}_{colormap.name}_{width}_{height}.mp4")  # type: ignore
            process = process.output(
                output_fullpath,
                preset='slow',
                movflags='faststart',
                pix_fmt='yuv420p',
                **{
                    'c:v': 'libx264', 'profile:v': 'high'
                },
            ).overwrite_output().run_async(
                pipe_stdin=True, quiet=True
            )

        all_times = []
        for i in range(nb_iter_done):
            start_time = time.time()

            img = get_image(np_all_cells[i], render_params['pixel_size'], colormap)
            process.stdin.write(img.tobytes())

            all_times.append(time.time() - start_time)
        process.stdin.close()
        process.wait()

        total_time = np.sum(all_times)
        mean_time = np.mean(all_times)
        print(f"{len(all_times)} images dumped in {total_time} seconds: {1. / mean_time} fps")

        all_outputs_fullpath.append(output_fullpath)

    return all_outputs_fullpath


def dump_gif(video_fullpath):
    """
        Command run:
            ffmpeg -i video_fullpath
                -vf "fps=10,scale=width:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
                -loop 0
                output.gif
    """
    output_fullpath = os.path.splitext(video_fullpath)[0] + '.gif'

    probe = ffmpeg.probe(video_fullpath)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])

    split = ffmpeg.input(video_fullpath).filter('scale', width, -1, flags='lanczos').filter('fps', fps=24).split()

    palette = split[0].filter('palettegen')

    ffmpeg_cmd = ffmpeg.filter([split[1], palette], 'paletteuse').output(output_fullpath).overwrite_output()

    ffmpeg_cmd.run(quiet=True)


def dump_qd_ribs_result(output_fullpath):
    """
        Command run:
            ffmpeg  -framerate 16 -i '%4d-emitter_0.png' \
                -framerate 16 -i '%4d-emitter_1.png' \
                -framerate 16 -i '%4d-emitter_2.png' \
                -framerate 16 -i '%4d-emitter_3.png' \
                -framerate 16 -i '%4d-archive_ccdf.png' \
                -framerate 16 -i '%4d-archive_heatmap.png' \
                -filter_complex "[0:v][1:v]hstack[h1];\
                    [2:v][3:v]hstack[h2];\
                    [4:v][5:v]hstack[h3];\
                    [h1][h2]vstack[v1];\
                    [v1][h3]vstack[o]"\
                -map "[o]"\
                out.mp4
    """
    inputs = [
        '%4d-emitter_0.png',
        '%4d-emitter_1.png',
        '%4d-emitter_2.png',
        '%4d-emitter_3.png',
        '%4d-archive_ccdf.png',
        '%4d-archive_heatmap.png',
    ]
    ffmpeg_inputs = []
    for i_str in inputs:
        ffmpeg_inputs.append(ffmpeg.input(i_str, framerate=10))
    h1 = ffmpeg.filter(ffmpeg_inputs[:2], 'hstack')
    h2 = ffmpeg.filter(ffmpeg_inputs[2:4], 'hstack')
    h3 = ffmpeg.filter(ffmpeg_inputs[4:6], 'hstack')
    v1 = ffmpeg.filter([h1, h2], 'vstack')
    v2 = ffmpeg.filter([v1, h3], 'vstack')
    ffmpeg_cmd = v2.output(
        output_fullpath, crf=20, preset='slower', movflags='faststart', pix_fmt='yuv420p'
    ).overwrite_output()

    ffmpeg_cmd.run(quiet=True)
