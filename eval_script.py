import json
# 你们修改了Output的结果
eval_path = "dataset/subgraph_kgp1_valid.json"
# 不需要修改
ground_truth_path = "dataset/subgraph_kgp1_valid.json"

with open(eval_path, 'r', encoding='utf-8') as f:
    eval_dict = json.load(f)

with open(ground_truth_path, 'r', encoding='utf-8') as f:
    label_dict = json.load(f)


for task in ['link_prediction', 'entity_prediction']:
    evals = eval_dict[task]
    labels = label_dict[task]
    total_scores = 0
    for eval, label in zip(evals, labels):
        eval_output = eval["output"]
        label_truth = set(label["ground_truth"])
        for idx, output in enumerate(eval_output):
            if output in label_truth:
                total_scores += 1.0/ (idx+1)

    print(task, total_scores/len(evals))

