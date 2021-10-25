import numpy as np


def hms(secs):
    '''Returns time in ``hour minute seconds`` fromat given time in seconds.'''
    h = secs//3600
    m = secs//60 - 60*h
    s = (secs - 60*m - 3600*h)//1
    return f'{h} h {m} m {s} s'
