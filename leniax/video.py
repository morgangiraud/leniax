import os
import time
import ffmpeg
import numpy as np

from . import utils as leniax_utils


def dump_video(save_dir, all_cells, render_params, colormap):
    assert len(all_cells.shape) == 4  # [nb_iter, C, H, W]

    nb_iter_done = len(all_cells)
    width = all_cells[0].shape[-1] * render_params['pixel_size']
    height = all_cells[0].shape[-2] * render_params['pixel_size']
    output_fullpath = os.path.join(save_dir, 'beast.mp4')
    process = (
        ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{width}x{height}", framerate=32
                     ).output(output_fullpath, crf=20, preset='slower', movflags='faststart',
                              pix_fmt='yuv420p').overwrite_output().run_async(pipe_stdin=True)
    )
    all_times = []

    for i in range(nb_iter_done):
        start_time = time.time()
        img = leniax_utils.get_image(
            all_cells[i], render_params['pixel_size'], render_params['pixel_border_size'], colormap
        )
        process.stdin.write(img.tobytes())

        all_times.append(time.time() - start_time)
    process.stdin.close()
    process.wait()

    total_time = np.sum(all_times)
    mean_time = np.mean(all_times)
    print(f"{len(all_times)} images dumped in {total_time} seconds: {mean_time} fps")


def dump_qd_ribs_result(output_fullpath):
    """
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
    ffmpeg.filter([v1, h3], 'vstack').output(
        output_fullpath, crf=20, preset='slower', movflags='faststart', pix_fmt='yuv420p'
    ).overwrite_output().run()
