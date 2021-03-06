import multiprocessing

# pip install tqdm -U
# autonotebook
#from tqdm.autonotebook import tqdm #  https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook-prints-new-progress-bars-repeatedly
from tqdm import tqdm

from functools import partial

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
# from multiprocessing.pool import ThreadPool as Pool
# https://github.com/baidut/VQA-DB/issues/58

# import time
import random

"""LOG
from multiprocessing import Pool; Pool.__name__
from multiprocessing.pool import ThreadPool; ThreadPool.__name__


* add parital support, *args and **kwargs

# %%
a = 2, 4,

a[2]


# %%
import ast
import numpy as np
df = pd.read_csv('D:/cmder/results.csv')
s=df.iloc[0]['f_brightness']
s
ast.literal_eval(s)
df['f_x']
s = df.iloc[0]['f_x'];ast.literal_eval(s)
s = df.iloc[0]['f_t'];ast.literal_eval(s)

# %%
"""

def try_func(func, x, *args, **kwargs):
    try:
        # choice = random.choice([0,1,2])
        # if choice == 0:
        #     raise NotImplementedError("NotImplemented!")
        # elif choice == 1:
        #     raise ValueError("Value!")
        #pbar.write(f'\r\n{x}')
        return x, func(x, *args, **kwargs) # if no return value, it will return x, None

    except Exception as e:
        # pbar = tqdm(total=0, position=0, bar_format='{desc}')
        print_msg = lambda msg: tqdm.write(f'\nERROR[ {x} ]: {msg}')

        if hasattr(e, 'message'):
            print_msg(e.message)
        else:
            print_msg(e)
        return None, {}

        """
        import pandas as pd; pd.DataFrame([{'a':1, 'b':2}, {'a':1, 'b':2}, {}])
        import pandas as pd; pd.DataFrame.from_dict([[1,2,3],[1,2,3] , [], {}])
        """

# update tqdm
"""
# %%
from parallel import *

par(print, list(range(5)), num_pool=0)
# %%
"""
def par(func, items, num_pool=None,
        unordered=False, thread=False, *args, **kwargs):
    num_cpu = multiprocessing.cpu_count()

    if num_pool == None:
        num_pool = num_cpu-1 if thread is False else 100

    if thread is False: # multiprocessing
        assert num_pool <= num_cpu, 'For multiprocessing, n_process <= n_cores otherwise processes will compete for CPUs'

    status = 'enabled' if num_pool >=1 else 'disabled'
    pool_type = ThreadPool if thread else Pool
    print(f'Parallel is {status}: pool_type = {pool_type.__name__}, num_pool = {num_pool}, num_cpu = {num_cpu}\n')

    warp_func = partial(try_func, func, *args, **kwargs)
    max_ = len(items)

    if num_pool > 1:
        with pool_type(num_pool) as p:
            if unordered:
                items_ = p.imap_unordered(warp_func, items)
            else:
                items_ = p.imap(warp_func, items)
            pbar = tqdm(items_, total=max_)
            res = []
            for x, y in pbar:
                pbar.set_description_str(f'{str(x)[:20]:<20}')
                if y is not {}:
                    res.append(y)

    else:
        pbar = tqdm(items, total=max_) # , position=2
        res = []
        for t in pbar:
            # time.sleep(random.uniform(0, 3))
            pbar.set_description_str(f'{str(t)[:20]:<20}') # set_postfix_str
            x, y = warp_func(t)
            if y is not {}:
                res.append(y)
    # summary
    print(f'{len(res)} out of {max_} succeeded.')
    return res
