import cv2
import numpy as np
import os

def mediaProcess(path, config):
    data = config.get('data')
    for inp, out in zip(data.get('input'), data.get('output')):
        oped = os.path.join(path, inp)
        for op in config.get('operation'):
            para = ''
            operator = op[0]
            for p in op[1:]:
                para += ',' + str(p)
            oped = eval("%s(oped%s)" % (operator, para))

        cv2.imwrite(os.path.join(path, out), oped)

    print(config.get('name') + " Finished.")
