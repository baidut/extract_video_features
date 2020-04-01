# from multiprocessing.pool import ThreadPool as Pool
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool
# from joblib import Parallel, delayed

# LOG = tqdm(total=0, position=1, bar_format='{desc}')

from functools import partial
# https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function

# def try_func(x, func, log_bar, msg_func, err_func, ok_func):
#     # file_log_bar.set_description_str(msg_func(x))
#     try:
#         if ok_func is None:
#             return func(x, log_bar=log_bar)
#         else:
#             return ok_func(func(x, log_bar=log_bar))
#
#     except Exception as e:
#         if err_func is None:
#             print(x)
#             raise e
#         else:
#             return err_func(x, e)

def try_func(func, x, log_bar):
    try:
        func(x, log_bar=log_bar)
        return True
    except Exception as e:
        raise e
        return False

# update tqdm
def par(items, func, num_pool,
        unordered=False,
        err_func=lambda x, e: 0,
        ok_func=lambda x: 1, # Use 0/1 instead of False/True: TypeError: cannot unpack non-iterable bool object
        msg_func=lambda x: str(x)):
    # def processInput(x):
    #     # sys.stdout = open(os.devnull, 'w')
    #     # warnings.filterwarnings("ignore")
    #     try:
    #         func(x)
    #         return 1
    #     except Exception as e:
    #         print(e)
    #         err_func(x)
    #         return 0
    #

    status = 'enabled' if num_pool >=1 else 'disabled'
    print(f'Parallel is {status}: num_pool = {num_pool}')

    log_bar = tqdm(total=0, position=1, bar_format='{desc}')
    # warp_func = partial(try_func, func, None, msg_func, err_func, ok_func)
    warp_func = partial(try_func, func, log_bar=log_bar)

    if num_pool >= 1:
        if num_pool == 1:
            num_pool = multiprocessing.cpu_count()

        if unordered:
            # current file information?
            # don't support lambda or local functions
            # r = list(tqdm(ximap_unordered(func, items, processes=num_pool)))
            # r = list(tqdm(p.map(func, items)))
            with Pool(num_pool) as p:
                r = list(tqdm(p.imap_unordered(warp_func, items)))
        else:
            raise NotImplementedError
            #r = list(tqdm(ximap(func, items, processes=num_pool)))

        # a = Parallel(n_jobs=num_pool)(delayed(processInput)(i) for i in pbar)

        # num_cpu = multiprocessing.cpu_count()
        # num_pool = min(num_pool, num_cpu)
        # print(f'num_cpu={num_cpu}, num_pool={num_pool}')
        # with Pool(num_pool) as p:  # restrict pool num to max core number
        #     # with tqdm(total=)
        #     results = t.map(func, items) # results = [[ret,num], [ret, num], ...]
        # t.close()
        # t.join()
        # pbar = tqdm(results)
        # for res in pbar:
        #     pbar.set_description(msg_func(res))

        # TODO: num_pool separated bars for each process
    else:
        pbar = tqdm(items)
        r = [warp_func(x) for x in pbar]

    return r #np.array(a)


"""
AttributeError: Can't pickle local object 'par.<locals>.
"""
