import multiprocessing
from tqdm import tqdm
from functools import partial

from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool as Pool

# import time
import random

"""
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

def try_func(func, x):
    try:
        # choice = random.choice([0,1,2])
        # if choice == 0:
        #     raise NotImplementedError("NotImplemented!")
        # elif choice == 1:
        #     raise ValueError("Value!")
        #pbar.write(f'\r\n{x}')
        return x, func(x) # if no return value, it will return x, None

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
def par(func, items, num_pool=None,
        unordered=False):
    if num_pool == None:
        num_pool = multiprocessing.cpu_count()

    status = 'enabled' if num_pool >=1 else 'disabled'
    print(f'Parallel is {status}: num_pool = {num_pool}\n')

    warp_func = partial(try_func, func)
    max_ = len(items)

    if num_pool > 1:
        with Pool(num_pool) as p:
            if unordered:
                items_ = p.imap_unordered(warp_func, items)
            else:
                items_ = p.imap(warp_func, items)
            pbar = tqdm(items_, total=max_)
            res = []
            for x, y in pbar:
                pbar.set_description(str(x))
                if y is not {}:
                    res.append(y)

    else:
        pbar = tqdm(items, total=max_) # , position=2
        res = []
        for t in pbar:
            # time.sleep(random.uniform(0, 3))
            pbar.set_description(str(t))
            x, y = warp_func(t)
            if y is not {}:
                res.append(y)
    # summary
    print(f'{len(res)} out of {max_} succeeded.')
    return res
