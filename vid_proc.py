import cv2

# crop
import random
import subprocess

from pathlib import Path
from get_features import *

"""
Feature list:


1. Absolute Luminance (1 per frame) : mean and sd
over all frames - 2 per video

2. RMS Luminance Contrast (1 per frame) : mean and
sd over all frames - 2 per video

3. Colorfulness (1 per frame) : mean and sd over all
frames - 2 per video

4. Number of faces (1 per frame) : mean and sd over
all frames - 2 per video

5. Directional Gaussian Derivatives (Spatial) -
3 scales, 2 orientations (x and y) -
spatial average (3x2 = 6 per frame) : mean and sd
over all frames - 12 per video

6. Directional Gaussian Derivatives (temporal) - 3
scales - temporal average (3 per pixel) : mean and sd
over all pixels - 6 per video

Overall: 26 per video
"""

def get_vid_length(vid_file):
    """
    # %%
    from vid_proc import get_vid_length
    get_vid_length('D:/temp_videos/WIN_20200323_17_43_22_Pro.mp4')
    # %%
    """
    # assume file exists
    cap = cv2.VideoCapture(str(vid_file))

    # check corrupted video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps is 0 or fps > 120 or \
        frameCount == 0 or frameWidth == 0 or frameHeight == 0:
        print(f'Not found (wrong path) or corrupted: {vid_file} ({frameWidth}x{frameHeight}) frameCount={frameCount}, ')
        # f.rename(Path('Corrupted') / f.name) # mkdir
        return -1
    cap.release()
    return frameCount/fps # how many seconds


def rand_crop_vid(vid_file, out_dir, duration): # 5 seconds
    """
    # %%
    from vid_proc import rand_crop_vid
    rand_crop_vid('D:/temp_videos/WIN_20200323_17_43_22_Pro.mp4', 'out')
    # %%
    """
    # get video length
    length = get_vid_length(vid_file)
    crop_start_sec = random.randint(0, int(length) - duration)
    # save cropped file at current folder
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cropped_file = Path(out_dir)/Path(vid_file).name # keep the original ext
    cmd = 'ffmpeg -y -v error -err_detect explode ' \
          f'-ss {crop_start_sec} -t {duration} ' \
          f'-i "{vid_file}" -c copy -an -avoid_negative_ts ' \
          f'make_zero "{cropped_file}"'
    subprocess.call(cmd, shell=True) # call will wait it finish
    return cropped_file

def get_vid_feats(f): # frame by frame
    #  spatial_feats=None, temporal_feats=None
    # if spatial_feats is None:
    #     spatial_feats = [get_brightness, get_contrast, get_colorfulness, get_face_num, get_gaussian_spatial]
    # if temporal_feats is None:
    #     temporal_feats = [get_gaussian_temporal]

    # assume file exists
    cap = cv2.VideoCapture(str(f))

    # check corrupted video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps is 0 or fps > 120 or \
       frameCount == 0 or frameWidth == 0 or frameHeight == 0:
        print(f'Corrupted: {f} ({frameWidth}x{frameHeight}) frameCount={frameCount}, ')
        # f.rename(Path('Corrupted') / f.name) # mkdir
        return

    # rows = []
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if not ret: break
    #     rows.append(tuple([func(frame) for func in funcs]))

    V = np.empty((frameCount, frameHeight, frameWidth), np.dtype("uint8"))
    # V = np.empty((frameCount, 200, 200), np.dtype('uint8'))

    c = 0
    ret = True

    f_brightness = []
    f_contrast = []
    f_colorfulness = []
    f_numFaces = []
    fx = []
    fy = []

    while c < frameCount and ret:
        ret, fr = cap.read()
        try:
            f_colorfulness.append(get_colorfulness(fr))
            fr = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        except BaseException:
            V[c] = np.zeros((frameHeight, frameWidth), dtype="uint8")
            c += 1
            return 0

        fx_temp, fy_temp = get_gaussian_spatial(fr)
        fx.append(fx_temp)
        fy.append(fy_temp)
        # add other features here
        f_brightness.append(get_brightness(fr))
        f_contrast.append(get_contrast(fr))
        f_numFaces.append(get_face_num(fr))

        # fr = cv2.resize(fr, dsize=(200,200), interpolation=cv2.INTER_LANCZOS4)
        V[c] = fr
        c += 1

    # stats
    features = {}
    features['f_brightness'  ] = [np.mean(f_brightness  , axis=0), np.std(f_brightness  , axis=0)]
    features['f_contrast'    ] = [np.mean(f_contrast    , axis=0), np.std(f_contrast    , axis=0)]
    features['f_colorfulness'] = [np.mean(f_colorfulness, axis=0), np.std(f_colorfulness, axis=0)]
    features['f_numFaces'    ] = [np.mean(f_numFaces    , axis=0), np.std(f_numFaces    , axis=0)]
    features['f_x'           ] = [np.mean(fx, axis=0).tolist(), np.std(fx, axis=0).tolist()]
    features['f_y'           ] = [np.mean(fy, axis=0).tolist(), np.std(fy, axis=0).tolist()]

    features['f_t'] = get_gaussian_temporal(V)

    cap.release()
    # two value in one feature (max and std)
    # feature_t 6 numbers
    return features
