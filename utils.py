import math
import random
import time

import torch

from data import lineToTensor


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def _randomChoice(random_value):
    return random_value[random.randint(0, len(l) - 1)]


def randomTrainingExample(all_categories, category_lines):
    category = _randomChoice(all_categories)
    line = _randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)
