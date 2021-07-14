import os
import time
import ffmpeg
import numpy as np

import lenia.utils as lenia_utils


def dump_video(all_cells, render_params, media_dir, colormap):
    assert len(all_cells.shape) == 6

    nb_iter_done = len(all_cells)
    width = all_cells[0].shape[-1] * render_params['pixel_size']
    height = all_cells[0].shape[-2] * render_params['pixel_size']
    process = (
        ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{width}x{height}").output(
            os.path.join(media_dir, 'beast.mp4'),
            pix_fmt='yuv420p',
        ).overwrite_output().run_async(pipe_stdin=True)
    )
    all_times = []
    for i in range(nb_iter_done):
        start_time = time.time()
        img = lenia_utils.get_image(
            all_cells[i][:, 0, 0, ...], render_params['pixel_size'], render_params['pixel_border_size'], colormap
        )
        process.stdin.write(img.tobytes())

        all_times.append(time.time() - start_time)
    process.stdin.close()
    process.wait()

    total_time = np.sum(all_times)
    mean_time = np.mean(all_times)
    print(f"{len(all_times)} images dumped in {total_time} seconds: {mean_time} fps")
