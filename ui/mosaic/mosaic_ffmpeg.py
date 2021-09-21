import os
import math

cdir = os.path.dirname(os.path.realpath(__file__))
collection_dir = os.path.join(cdir, '..', '..', 'outputs', 'collection-01')
config_filename = 'config.yaml'

for mp4_size in [128, 256, 512]:
    all_gifs = {}
    for (subdir, dirs, filenames) in os.walk(collection_dir):
        dirs.sort()

        mp4_filename = ''
        for filename in filenames:
            if 'creature' in filename and 'mp4' in filename and str(mp4_size) in filename:
                mp4_filename = filename
                break

        if mp4_filename == '':
            continue
        config_fullpath = os.path.join(subdir, config_filename)
        if not os.path.isfile(config_fullpath):
            continue

        family_dir_name = subdir.split('/')[-2]
        if family_dir_name not in all_gifs:
            all_gifs[family_dir_name] = []

        all_gifs[family_dir_name].append(os.path.join(subdir, mp4_filename))

    for family_dir_name, family_mp4 in all_gifs.items():
        # if int(family_dir_name.split('-')[0]) > 0:
        #     continue
        nb_mp4 = len(family_mp4)

        grid_width = int(math.sqrt(nb_mp4))
        if grid_width**2 < nb_mp4:
            if grid_width * (grid_width + 1) < nb_mp4:
                grid_width += 1
                grid_height = grid_width
            else:
                grid_height = grid_width + 1
        else:
            grid_height = grid_width

        input_videos = ""
        input_setpts = "nullsrc=size={}x{} [base];".format(mp4_size * grid_width, mp4_size * grid_height)
        input_overlays = ""

        for index, path_video in enumerate(family_mp4):
            input_videos += " -i " + path_video
            input_setpts += "[{}:v] setpts=PTS-STARTPTS, scale={}x{} [video{}];".format(
                index, mp4_size, mp4_size, index
            )
            text = "drawtext=text='{}':x={}:y={}:fontfile='/Library/Fonts/Arial Unicode.ttf':fontsize={}:fontcolor=white".format(  # noqa: E501
                path_video.split('/')[-2],
                mp4_size * (index % grid_width),
                mp4_size * (index // grid_width),
                mp4_size // 8,
            )
            if index == 0:
                input_overlays += "[base][video{}] overlay=shortest=1 [tmp{}];".format(index, index)
                input_overlays += "[tmp{}]{}[tmp{}];".format(index, text, index)
            elif index > 0 and index < len(family_mp4) - 1:
                input_overlays += "[tmp{}][video{}] overlay=shortest=1:x={}:y={} [tmp{}];".format(
                    index - 1, index, mp4_size * (index % grid_width), mp4_size * (index // grid_width), index
                )
                input_overlays += "[tmp{}]{}[tmp{}];".format(index, text, index)
            else:
                input_overlays += "[tmp{}][video{}] overlay=shortest=1:x={}:y={} [tmp{}];".format(
                    index - 1, index, mp4_size * (index % grid_width), mp4_size * (index // grid_width), index
                )
                input_overlays += "[tmp{}]{}".format(index, text)

        video_fullpath = os.path.join(cdir, f"{family_dir_name}-{mp4_size}.mp4")
        complete_command = f"ffmpeg -y{input_videos} -filter_complex \"{input_setpts}{input_overlays}\" -c:v libx264 {video_fullpath}"  # noqa: E501
        os.system(complete_command)
