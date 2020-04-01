"""
!python D:/Git/extract_video_features/main.py --help
!python D:/Git/extract_video_features/main.py hello --name Bob --count 3
!python "D:/Git/extract_video_features/main.py" test-vid "D:/temp_videos/WIN_20200323_17_43_22_Pro.mp4"
!python "D:/Git/extract_video_features/main.py" test-vid  --duration 1 "D:/temp_videos/WIN_20200323_17_43_22_Pro.mp4"
!python "D:/Git/extract_video_features/main.py" extract-dir  --duration 1 "D:/temp_videos"
python "D:/Git/extract_video_features/main.py" extract-dir -o "D:/Git/extract_video_features/output5" -p 4 --duration 1 "D:/temp_videos/"
python "/mnt/d/Git/extract_video_features/main.py" extract-dir -o "/mnt/d/Git/extract_video_features/output5" -p 4 --duration 1 "/mnt/d/temp_videos/"
"""

import click
import numpy as np
import pandas as pd
import sys
import json

from parallel import * # run in parallel
from vid_proc import rand_crop_vid, get_vid_feats

from tqdm import tqdm
from functools import partial

# %%
from pathlib import Path


def extract_features(vid_file, out_dir, duration, skip_exist=True, log_bar=None):
    """
    # %%
    from main import *
    # extract_features('D:/temp_videos/WIN_20200323_17_43_22_Pro.mp4', 'D:/temp_videos/output')
    extract_features('D:/temp_videos/WIN_20200323_17_55_12_Pro.mp4', 'D:/temp_videos/output')
    extract_features('D:/temp_videos/WIN_20200323_17_55_12_Pro.mp4', 'out')
    # %%
    """
    print_msg = lambda msg: print(msg) if log_bar is None else log_bar.set_description_str(msg)

    result_file = Path(out_dir)/f'{Path(vid_file).stem}.txt'
    if skip_exist and result_file.exists():
        print_msg(f'Skipped existed: {vid_file}')
        return

    print_msg(f'Cropping {vid_file}...')
    cropped_file = rand_crop_vid(vid_file, out_dir, duration)
    print_msg(f'Computing features {vid_file}...')
    feats = get_vid_feats(cropped_file)
    cropped_file.unlink() # delete the cropped_file
    with open(result_file, 'w') as json_file:
        json.dump(feats, json_file)
    return result_file

@click.group()
def cli():
    pass

@cli.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo('Hello %s!' % name)


@cli.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hi(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo('Hi %s!' % name)

@cli.command()
@click.option('--out_dir', default='out')
@click.option('--duration', default=5)
@click.argument('file')
def test_vid(file, out_dir, duration):
    print(f'Extracting [{file}] --> [{out_dir}] ...')
    result_file = extract_features(file, out_dir, duration)
    print('DONE')
    with open(result_file) as json_file:
        data = json.load(json_file)
        print(data)

# LOG = tqdm(total=0, position=1, bar_format='{desc}')
# def func(x):
#     LOG.set_description_str(str(x))
#     try:
#         # func(x, log=LOG)
#         extract_features(x, out_dir, duration, log=LOG)
#         return 1
#     except Exception as e:
#         LOG.write(f'{x}: {e}')
#         return 0

# fuck = lambda f, **kw: extract_features(f, 'out_dir', 5, **kw)

@cli.command()
@click.option('--out_dir', '-o', default='out')
@click.option('--duration', default=5)
@click.option('--num_pool', '-p', default=12, type=int)
@click.argument('in_dir')
def extract_dir(in_dir, out_dir, duration, num_pool):
    # get video files
    VIDEO_EXTS = '.mp4', '.mpeg4', '.webm', '.mov', '.qt', '.avi', '.flv', '.wmv', \
        '.mkv', '.yuv', '.m4v', '.m4p', '.vob', '.ogv', '.ogg', '.rm', '.mpg', \
        '.mpeg', '.3gp', '.3g2', '.ts', '.asf'
    files = np.array([f for f in Path(in_dir).rglob('*.*') if f.name.lower().endswith(VIDEO_EXTS)])

    func = partial(extract_features, out_dir=out_dir, duration=duration)
    a = par(files, func, num_pool, unordered=True)
    # a = list(tqdm(xmap(func, files, processes=num_pool)))

    # _pickle.PicklingError: Could not pickle the task to send it to the workers.
    a = np.array(a)
    print(f'{a.sum()} out of {len(files)} succeeded')
    print('Checking failed files...')
    par(files[a==0], func, num_pool=0, err_func=None) # run failed files again


if __name__ == '__main__':
    cli()
