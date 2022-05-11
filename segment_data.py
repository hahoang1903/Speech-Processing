import os

from data import segment_sound

if not os.path.exists('data/segmented'):
    for subdir, dirs, _ in os.walk('data/raw'):
        for dir in dirs:
            segment_sound(os.path.join(subdir, dir))
