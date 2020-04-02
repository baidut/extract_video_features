"""
# %%
from vid_proc import get_vid_feats
%time get_vid_feats('output_dir/3179649855.mp4')
# %%
import time
import random

def myfunc(a):
    time.sleep(random.uniform(0, 1))


from parallel import *

# %%
#%time
par(myfunc, range(10), unordered=True)


!python main.py extract-dir --help

# %%
"""

import click
import numpy as np
import pandas as pd
import sys
import json

from parallel import * # run in parallel
from vid_proc import rand_crop_vid, get_vid_feats


# %%
from pathlib import Path


def extract_features(vid_file, out_dir, duration, skip_exist=True):
    #global PBAR
    print_msg = lambda x: x # print(x) # PBAR.set_description_str(str(x))
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
    if skip_exist and result_file.exists():
        print_msg(f'Skipped existed: {vid_file}')
    else:
        print_msg(f'Cropping {vid_file}...')
        cropped_file = rand_crop_vid(vid_file, out_dir, duration)
        print_msg(f'Computing features {vid_file}...')
        feats = get_vid_feats(cropped_file)
        cropped_file.unlink() # delete the cropped_file
        with open(result_file, 'w') as json_file:
            json.dump(feats, json_file)

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
@click.option('--duration', default=5, help="The duration of the cropped video (in seconds). "
"ffmpeg will start decoding at a random position and stop decoding after [duration] seconds. "
"Note the cropped video file will be slightly longer than the duration.")
@click.option('--num_pool', '-p', default=6, type=int, help="num of processes in parallel")
@click.argument('in_dir')
def extract_dir(in_dir, out_dir, duration, num_pool):
    """For each video under input directory [in_dir], the program will output
    the feature values to a ".json" file with the same filename under the output directory [out_dir]."""
    # get video files
    VIDEO_EXTS = '.mp4', '.mpeg4', '.webm', '.mov', '.qt', '.avi', '.flv', '.wmv', \
        '.mkv', '.yuv', '.m4v', '.m4p', '.vob', '.ogv', '.ogg', '.rm', '.mpg', \
        '.mpeg', '.3gp', '.3g2', '.ts', '.asf'
    files = np.array([f for f in Path(in_dir).rglob('*.*') if f.name.lower().endswith(VIDEO_EXTS)])
    a = par(extract_features, files, num_pool, unordered=True, out_dir=out_dir, duration=duration)

@cli.command()
@click.option('--num_pool', '-p', default=6, type=int)
@click.option('--csv_file', default='results.csv')
@click.argument('json_dir')
def join_results(json_dir, csv_file, num_pool):
    """Merge the json files under [json_dir] to a CSV file [csv_file]"""
    files = np.array([f for f in Path(json_dir).rglob('*.*') if f.name.lower().endswith('.json')])
    print(f'{len(files)} files to be joined' )
    df = pd.DataFrame(par(load_features, files, num_pool, unordered=True))
    df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    cli()

"""
import multiprocessing; multiprocessing.cpu_count()

%time !python main.py extract-dir -o "output_dir" --duration 1 -p 0 "input_dir"

from vid_proc import get_vid_length
get_vid_length('input_dir/2999049224.mp4')

from vid_proc import rand_crop_vid
rand_crop_vid('input_dir/2999049224.mp4', 'output_dir', duration=1)

from vid_proc import get_vid_feats
%time get_vid_feats('input_dir/3179649855.mp4')

from vid_proc import get_vid_feats
%time get_vid_feats('output_dir/2999049224.mp4')

from main import extract_features
extract_features('input_dir/2999049224.mp4', 'output_dir', duration=1)

# %%
from main import *
%time extract_features('input_dir/3240926995.mp4', out_dir='output_dir', duration=1)
# %%

from main import load_features
load_features('output_dir/2999049224.json')
"""
