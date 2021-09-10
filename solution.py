import math
import re
import sys


def unique_args(data, i):
    return list(set([row[i] for row in data]))


def class_count(rows, result_list):
    count = {}
    for r in result_list:
        count[r] = 0
    for row in rows:
        label = row[-1]
        count[label] += 1
    return count


def rows_for_arg(rows, arg, k):
    result_rows = []
    for row in rows:
        if arg == row[k]:
            result_rows.append(row)
    return result_rows


class Leaf:
    def __init__(self, v):
        self.v = v


class Node:
    def __init__(self, x, subtrees, w):
        self.x = x
        self.subtrees = subtrees
        self.w = w


def print_tree(node, depth):
    global print_var
    if isinstance(node, Leaf):
        return
    print_var += str(depth) + ':' + node.x + ", "
    for n in node.subtrees:
        print_tree(n[1], depth + 1)


def find_entropy(count, y, n):
    E = 0
    for j in y:
        m = count[j]
        if m == 0:
            continue
        else:
            log = math.log2(m / n)
        E += (m / n) * log
    return -E


def argmax_IG(rows, X, y, data_dict, headers):
    global mode
    count = class_count(rows, y)
    e = find_entropy(count, y, len(rows))
    save_e = e
    ig_dict = {}
    for x in X:
        e = save_e
        for data in data_dict[x]:
            count = class_count(rows_for_arg(rows, data, headers.index(x)), y)
            e -= (len(rows_for_arg(rows, data, headers.index(x))) / len(rows)) * find_entropy(count, y, len(
                rows_for_arg(rows, data, headers.index(x))))
        ig_dict.update({x: e})
    max_ig = -100
    for key in sorted(ig_dict):
        if ig_dict.get(key) > max_ig:
            max_key = key
            max_ig = ig_dict.get(key)
        if mode != 'test':
            print('IG(' + key + ')=' + str(ig_dict.get(key)))
    return max_key


def argmax_leaf(rows, y):
    count = class_count(rows, y)
    help_y = sorted(y)
    max_v = count[help_y[0]]
    max_key = help_y[0]
    for key in help_y:
        if count.get(key) > max_v:
            max_key = key
            max_v = count.get(key)
    return max_key


def count_v_rows(rows, v):
    result = 0
    for row in rows:
        if v == row[-1]:
            result += 1
    return result


def id3(rows, parent_rows, X, y, data_dict, target, headers, max_depth, depth):
    if not rows:
        v = argmax_leaf(parent_rows, y)
        return Leaf(v)
    w = argmax_leaf(rows, y)
    if not X or len(rows) == count_v_rows(rows, w) or depth == max_depth:
        return Leaf(w)
    x = argmax_IG(rows, X, data_dict[target], data_dict, headers)
    subtrees = []
    for v in data_dict[x]:
        help_X = list(X)
        help_X.remove(x)
        t = id3(rows_for_arg(rows, v, headers.index(x)), rows, help_X, y, data_dict, target, headers, max_depth,
                depth + 1)
        subtrees = subtrees + [(v, t)]
    return Node(x, subtrees, w)


def make_matrix_dict_key(predicted, real):
    return 'predicted(' + predicted + ')->real(' + real + ')'


def prediction(node, row, headers):
    global print_predictions
    global matrix_dict
    if isinstance(node, Leaf):
        print_predictions += node.v + " "
        matrix_dict[make_matrix_dict_key(node.v, row[-1])] += 1
        return node.v == row[-1]
    k = headers.index(node.x)
    feature = row[k]
    for subtree in node.subtrees:
        if subtree[0] == feature:
            return prediction(subtree[1], row, headers)
    return node.w


def predict_test_data(tree, rows, headers):
    global print_predictions
    correct = 0
    total = len(rows)
    for row in rows:
        if prediction(tree, row, headers):
            correct += 1
    print(print_predictions)
    print("%.5f" % (round(correct / total, 5)))


class ID3:

    def fit(self, rows, X, y, data_dict, target, headers, max_depth, depth):
        self.tree = id3(rows, rows, X, y, data_dict, target, headers, max_depth, depth)
        print_tree(self.tree, 0)

    def predict(self, rows, headers):
        predict_test_data(self.tree, rows, headers)


def read_from_file(path):
    with open(path, encoding='utf8') as datafile:
        lines = datafile.read().split('\n')
        lines = [x for x in lines if x]
        return lines


datainputlines = read_from_file(sys.argv[1])
testinputlines = read_from_file(sys.argv[2])

header_list = datainputlines.pop(0).split(',')
test_header_list = testinputlines.pop(0).split(',')
features = []
test_data = []
for i in datainputlines:
    features.append(i.split(','))
for i in testinputlines:
    test_data.append(i.split(','))

with open(sys.argv[3]) as configfile:
    configinputlines = configfile.read()
mode = re.search('mode.*\n', configinputlines).group().split('=')[1]
mode = mode.strip('\n')
model = re.search('model.*\n', configinputlines).group().split('=')[1]
model = model.strip('\n')
max_depth = re.search('max_depth.*\n', configinputlines).group()
max_depth = -1 if max_depth is None else int(max_depth.split('=')[1])
num_trees = re.search('num_trees.*\n', configinputlines).group()
num_trees = 1 if num_trees is None else int(num_trees.split('=')[1])
feature_ratio = re.search('feature_ratio.*\n', configinputlines).group()
feature_ratio = 1 if feature_ratio is None else float(feature_ratio.split('=')[1])
example_ratio = re.search('example_ratio.*\n', configinputlines).group()
example_ratio = 1 if example_ratio is None else float(example_ratio.split('=')[1])

unique_data = {}
for i in range(len(header_list)):
    unique_data.update({header_list[i]: unique_args(features, i)})

print_var = ''
id3_model = ID3()
target = unique_data[header_list[-1]]
id3_model.fit(features, header_list[0:-1], target, unique_data, header_list[-1],
              header_list, max_depth, 0)
print_var = print_var[0:-2]
print(print_var)
print_predictions = ''
matrix_dict = {}
for y1 in target:
    for y2 in target:
        matrix_dict[make_matrix_dict_key(y1, y2)] = 0
id3_model.predict(test_data, test_header_list)
for y1 in sorted(target):
    row_print = ''
    for y2 in sorted(target):
        row_print += str(matrix_dict[make_matrix_dict_key(y2, y1)]) + ' '
    print(row_print)
