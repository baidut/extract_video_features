"""
!python D:/Git/extract_video_features/main.py --help
!python D:/Git/extract_video_features/main.py hello --name Bob --count 3
python /mnt/d/Git/extract_video_features/main.py hello --name Bob --count 3
# tqdm opencv-python numpy
!python "D:/Git/extract_video_features/main.py" test-vid "D:/temp_videos/WIN_20200323_17_43_22_Pro.mp4"
!python "D:/Git/extract_video_features/main.py" test-vid  --duration 1 "D:/temp_videos/WIN_20200323_17_43_22_Pro.mp4"
!python "D:/Git/extract_video_features/main.py" extract-dir  --duration 1 "D:/temp_videos"
python "D:/Git/extract_video_features/main.py" extract-dir -o "D:/Git/extract_video_features/output5" -p 4 --duration 1 "D:/temp_videos/"
python "/mnt/d/Git/extract_video_features/main.py" extract-dir -o "/mnt/d/Git/extract_video_features/output5" -p 4 --duration 1 "/mnt/d/temp_videos/"
python "D:/Git/extract_video_features/main.py" join-results -p 4 "D:/Git/extract_video_features/output"
"""

"""
# %%
import time
import random

def myfunc(a):
    time.sleep(random.uniform(0, 1))


from parallel import *

# %%
#%time
par(myfunc, range(10), unordered=True)

# %%
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


def extract_features(vid_file, out_dir, duration, skip_exist=True):
    #global PBAR
    print_msg = lambda x: x # PBAR.set_description_str(str(x))
    """
    # %%
    from main import *
    # extract_features('D:/temp_videos/WIN_20200323_17_43_22_Pro.mp4', 'D:/temp_videos/output')
    extract_features('D:/temp_videos/WIN_20200323_17_55_12_Pro.mp4', 'D:/temp_videos/output')
    extract_features('D:/temp_videos/WIN_20200323_17_55_12_Pro.mp4', 'out')
    # %%
    """
    # print_msg = lambda msg: print(msg) if log_bar is None else log_bar.set_description_str(msg)

    result_file = Path(out_dir)/f'{Path(vid_file).stem}.json'
    if not skip_exist and result_file.exists():
        print_msg(f'Cropping {vid_file}...')
        cropped_file = rand_crop_vid(vid_file, out_dir, duration)
        print_msg(f'Computing features {vid_file}...')
        feats = get_vid_feats(cropped_file)
        cropped_file.unlink() # delete the cropped_file
        with open(result_file, 'w') as json_file:
            json.dump(feats, json_file)
    else:
        print_msg(f'Skipped existed: {vid_file}')

    return result_file

def load_features(result_file):
    with open(str(result_file)) as json_file:
        data = json.load(json_file)
        data['file'] = Path(result_file).stem
        return data

@click.group()
def cli():
    pass

# def my_func(x):
#     time.sleep(random.uniform(0, 1))
#
# @cli.command()
# @click.option('--num_pool', '-p', default=0, type=int)
# def testpar(num_pool):
#     par(my_func, range(10), num_pool=num_pool, unordered=True)


@cli.command()
@click.option('--out_dir', prompt='Please specify the output folder')
@click.option('--duration', default=5)
@click.argument('file')
def test_vid(file, out_dir, duration):
    print(f'Extracting [{file}] --> [{out_dir}] ...')
    result_file = extract_features(file, out_dir, duration)
    print('DONE')
    print(load_features(result_file))

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
@click.option('--out_dir', '-o', prompt='Please specify the output folder')
@click.option('--duration', default=5)
@click.option('--num_pool', '-p', default=6, type=int)
@click.argument('in_dir')
def extract_dir(in_dir, out_dir, duration, num_pool):
    # get video files
    VIDEO_EXTS = '.mp4', '.mpeg4', '.webm', '.mov', '.qt', '.avi', '.flv', '.wmv', \
        '.mkv', '.yuv', '.m4v', '.m4p', '.vob', '.ogv', '.ogg', '.rm', '.mpg', \
        '.mpeg', '.3gp', '.3g2', '.ts', '.asf'
    files = np.array([f for f in Path(in_dir).rglob('*.*') if f.name.lower().endswith(VIDEO_EXTS)])

    func = partial(extract_features, out_dir=out_dir, duration=duration)
    a = par(func, files, num_pool, unordered=True)

@cli.command()
@click.option('--num_pool', '-p', default=6, type=int)
@click.option('--csv_file', default='results.csv')
@click.argument('json_dir')
def join_results(json_dir, csv_file, num_pool):
    files = np.array([f for f in Path(json_dir).rglob('*.*') if f.name.lower().endswith('.json')])
    print(f'{len(files)} files to be joined' )
    df = pd.DataFrame(par(load_features, files, num_pool, unordered=True))
    df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    cli()
