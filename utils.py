import os, logging
import sys
import time
import math
import numpy as np
import torch

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 86.

last_time = time.time()
begin_time = last_time

def set_logging_defaults(logdir, args):
    if os.path.isdir(logdir):
        # TODO
        pass #overwrite
        # res = input('"{}" exists. Overwrite [Y/n]? '.format(logdir))
        # if res != 'Y':
        #     raise Exception('"{}" exists.'.format(logdir))
    else:
        os.makedirs(logdir)

    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])

    # log cmdline argumetns
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)


    for mode in ['train', 'val']:
        logger = logging.getLogger(mode)
        logger.addHandler(logging.FileHandler(os.path.join(logdir,mode+'.txt')))

    


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# def one_hot_embedding(labels, num_classes):
#     """Embedding labels to one-hot form.

#     Args:
#       labels: (LongTensor) class labels, sized [N,].
#       num_classes: (int) number of classes.

#     Returns:
#       (tensor) encoded labels, sized [N, #classes].
#     """

#     # print(labels, type(labels))
#     # print(labels, type(labels))
#     y = torch.eye(num_classes) 
#     a=y[labels]
    
#     return a






    
        

def get_sample_idx_list(length, num_sampled):
    interval = length//num_sampled
    offset = np.random.randint(interval)
    sampled_idx_list = [interval*idx+offset for idx in range(num_sampled)]
    return sampled_idx_list



def make_batch(samples):
    inputs = [sample[0] for sample in samples]
    actions = [sample[1] for sample in samples]
    moments = [sample[2] for sample in samples]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return {
        'frame_seq': padded_inputs.contiguous(),
        'action': torch.stack(actions).contiguous(),
        'moment': torch.stack(moments).contiguous()
    }


