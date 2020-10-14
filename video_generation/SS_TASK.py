from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict
import random
# self supervised task
random.seed(1)
SS_TASK = list()

positive_ratio_set = list([x/10. for x in range(3, 8)])
task_skeleton_set = {'pos': [], 'neg': []}

schedule_set = list()

schedule = edict()
schedule.pos = [[0,180,0]]
schedule.neg = [[0,45,0], [0,90,0], [0,179]]
# schedule.neg = [[0,-180],[0,-180,0], [0,90,0],[0,179,350], [0,179]]
# schedule.neg = [[0,10,0]]
schedule_set.append(schedule)

# schedule = edict()
# schedule.pos = [[0,90,0], [5,95,5]]
# schedule.neg = [[0,90], [0,-90], [0,50,0], [0,90,180], [0,150,60], [0, -90, 0]]
# schedule_set.append(schedule)

# schedule = edict()
# schedule.pos = [[0,90]]
# schedule.neg = [[0,50], [0,-90], [0,-50], [0,30,-30]]
# schedule_set.append(schedule)




def generate_rotation_task():
	rotation_task = edict()
	rotation_task.positive_ratio = random.choice(positive_ratio_set)
	schedule = random.choice(schedule_set)
	rotation_task.pos = schedule.pos
	rotation_task.neg = schedule.neg

	return rotation_task


def sample_rotation_task():
	return random.choice(SS_TASK)

def get_rotation_task(idx):
	return SS_TASK[idx]



SS_TASK.append(generate_rotation_task())
SS_TASK.append(generate_rotation_task())
SS_TASK.append(generate_rotation_task())

