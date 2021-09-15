import os
import math

cdir = os.path.dirname(os.path.realpath(__file__))
collection_dir = os.path.join(cdir, '..', '..', 'outputs', 'collection-01')
config_filename = 'config.yaml'

mp4_size = 128
all_gifs =  {}
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

for family_dir_name, family_gifs in all_gifs.items():
    nb_gif = len(family_gifs)

    grid_width = int(math.sqrt(nb_gif))
    if grid_width**2 < nb_gif:
        grid_height = grid_width + 1
    else:
        grid_height = grid_width

    input_videos = ""
    input_setpts = "nullsrc=size={}x{} [base];".format(mp4_size * grid_width, mp4_size * grid_height)
    input_overlays = ""
    
    for index, path_video in enumerate(family_gifs):
        input_videos += " -i " + path_video
        input_setpts += "[{}:v] setpts=PTS-STARTPTS, scale={}x{} [video{}];".format(
            index, mp4_size, mp4_size, index
        )
        if index == 0:
            input_overlays += "[base][video{}] overlay=shortest=1 [tmp{}];".format(
                index, index
            )
            input_overlays += "[tmp{}]drawtext=text='{}':x={}:y={}:fontfile='/Library/Fonts/Arial Unicode.ttf':fontsize=12:fontcolor=white[tmp{}];".format(
                index, path_video.split('/')[-2], 0, 0, index
            )
        elif index > 0 and index < len(family_gifs) - 1:
            input_overlays += "[tmp{}][video{}] overlay=shortest=1:x={}:y={} [tmp{}];".format(
                index-1, index, mp4_size * (index % grid_width), mp4_size * (index//grid_width), index
            )
            input_overlays += "[tmp{}]drawtext=text='{}':x={}:y={}:fontfile='/Library/Fonts/Arial Unicode.ttf':fontsize=12:fontcolor=white[tmp{}];".format(
                index, path_video.split('/')[-2], mp4_size * (index % grid_width), mp4_size * (index//grid_width), index
            )
        else:
            input_overlays += "[tmp{}][video{}] overlay=shortest=1:x={}:y={} [tmp{}];".format(
                index-1, index, mp4_size * (index % grid_width), mp4_size * (index//grid_width), index
            )
            input_overlays += "[tmp{}]drawtext=text='{}':x={}:y={}:fontfile='/Library/Fonts/Arial Unicode.ttf':fontsize=12:fontcolor=white".format(
                index, path_video.split('/')[-2], mp4_size * (index % grid_width), mp4_size * (index//grid_width)
            )

    complete_command = "ffmpeg -y" + input_videos + " -filter_complex \"" + input_setpts + input_overlays + "\" -c:v libx264 " + os.path.join(cdir, family_dir_name) + ".mp4"
    os.system(complete_command)
