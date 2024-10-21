import codecs

# 创建一个全局三元组 set
triplet_set = set()

def load_train_data(file_path):
    train_file = file_path + "train.txt"
    entity2id_file = file_path + "entity2id.txt"
    relation2id_file = file_path + "relation2id.txt"

    entity2id = {}
    relation2id = {}

    # 读取 entity2id 文件
    with codecs.open(entity2id_file, 'r', encoding='utf-8') as f1:
        for line in f1:
            line = line.strip().split()  # 用空格分割
            if len(line) != 2:
                continue
            entity2id[line[0]] = int(line[1])  # 构建实体到id的字典

    # 读取 relation2id 文件
    with codecs.open(relation2id_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip().split()  # 用空格分割
            if len(line) != 2:
                continue
            relation2id[line[0]] = int(line[1])  # 构建关系到id的字典

    triple_list = []  # 用于存储三元组

    # 读取训练集文件
    with codecs.open(train_file, 'r', encoding='utf-8') as f:
        content = f.readlines()  # 读取所有行
        for line in content:
            triple = line.strip().split()  # 用空格分割
            if len(triple) != 3:
                continue

            h_ = entity2id.get(triple[0], None)  # 获取头实体的编号
            t_ = entity2id.get(triple[2], None)  # 获取尾实体的编号
            r_ = relation2id.get(triple[1], None)  # 获取关系的编号

            if h_ is not None and t_ is not None and r_ is not None:
                triple_list.append(triple)  # 储存三元组的编号
                triplet_set.add((h_, r_, t_))  # 将三元组的id加入全局set

    return triple_list, entity2id, relation2id


def load_test_data(file_path):
    test_file = file_path + "test.txt"
    entity2id_file = file_path + "entity2id.txt"
    relation2id_file = file_path + "relation2id.txt"

    entity2id = {}
    relation2id = {}

    # 读取 entity2id 文件
    with codecs.open(entity2id_file, 'r', encoding='utf-8') as f1:
        for line in f1:
            line = line.strip().split()  # 用空格分割
            if len(line) != 2:
                continue
            entity2id[line[0]] = int(line[1])  # 构建实体到id的字典

    # 读取 relation2id 文件
    with codecs.open(relation2id_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip().split()  # 用空格分割
            if len(line) != 2:
                continue
            relation2id[line[0]] = int(line[1])  # 构建关系到id的字典

    triple_list = []  # 用于存储三元组

    # 读取训练集文件
    with codecs.open(test_file, 'r', encoding='utf-8') as f:
        content = f.readlines()  # 读取所有行
        for line in content:
            triple = line.strip().split()  # 用空格分割
            if len(triple) != 3:
                continue

            h_ = entity2id.get(triple[0], None)  # 获取头实体的编号
            t_ = entity2id.get(triple[2], None)  # 获取尾实体的编号
            r_ = relation2id.get(triple[1], None)  # 获取关系的编号

            if h_ is not None and t_ is not None and r_ is not None:
                triple_list.append(triple)  # 储存三元组的编号
                triplet_set.add((h_, r_, t_))  # 将三元组的id加入全局set


    return triple_list, entity2id, relation2id