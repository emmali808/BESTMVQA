import json
import os
import random
import re

import pandas as pd

# from make_arrow import make_arrow, make_arrow_vqa, make_arrow_melinda

import re
import os
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

# from glossary import normalize_word


def statistics(iid2captions, iid2split):
    all_images = {"train": [], "val": [], "test": []}
    all_texts = {"train": [], "val": [], "test": []}

    for iid, texts in iid2captions.items():
        split = iid2split[iid]
        all_images[split].append(iid)
        all_texts[split].extend(texts)

    for split, images in all_images.items():
        print(f"+ {split} set: {len(images)} images")

    for split, texts in all_texts.items():
        lengths = [len(text.split()) for text in texts]
        avg_len = sum(lengths) / len(lengths)
        print(f"+ {split} set: {len(texts)} texts")
        print(f"+ {split} set: {avg_len} words in average.")
        lengths = [length // 10 * 10 for length in lengths]
        print(Counter(lengths))


def path2rest(path, iid2captions, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    import pdb
    pdb.set_trace()
    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_mimic_cxr(path, iid2captions, iid2chexpert, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    chexpert = iid2chexpert[name]
    split = iid2split[name]
    return [binary, captions, name, chexpert, split]


def make_arrow_mimic_cxr(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2chexpert = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2chexpert[sample["img_path"]].extend(sample["chexpert"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    import pdb
    pdb.set_trace()
    bs = [path2rest_mimic_cxr(path, iid2captions, iid2chexpert, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "chexpert", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def get_score(occurences):
    return 1.0


def path2rest_vqa(path, split, annotations, label2ans):
    with open(path, "rb") as fp:
        binary = fp.read()

    iid = path
    _annotation = annotations[split][iid]
    _annotation = list(_annotation.items())
    qids, qas = [a[0] for a in _annotation], [a[1] for a in _annotation]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas]
    answer_labels = [a["labels"] for a in answers]
    answer_scores = [a["scores"] for a in answers]
    question_types = [a["answer_type"] for a in answers]
    answers = [[label2ans[l] for l in al] for al in answer_labels]

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, question_types, split]


def make_arrow_vqa(data, dataset_name, save_dir):
    questions_test = data["test"]

    # Record Questions
    annotations = dict()
    for split, questions in zip(["test"], [questions_test]):
        _annotation = defaultdict(dict)
        for q in tqdm(questions):
            _annotation[q["img_path"]][q["qid"]] = [q["question"]]
        annotations[split] = _annotation

    # Construct Vocabulary
    all_major_answers = list()
    for split, questions in zip(["test"], [questions_test]):
        for q in tqdm(questions):
            all_major_answers.append(str(q["answer"]).lower())
    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())
    # later by hxj
    # save label2ans into json
    # import json
    # with open('/home/coder/projects/METER/data/vqa_rad/label2ans.json', 'w') as f:
    #     json.dump(label2ans, f)
    # print("Label size ({}): {}.".format(dataset_name, len(ans2label)))
    # print("########", len(label2ans))

    # Record Answers
    for split, questions in zip(["test"], [questions_test]):
        _annotation = annotations[split]
        for q in tqdm(questions):
            answers = normalize_word(str(q["answer"]).lower())
            answer_count = {}
            answer_count[answers] = answer_count.get(answers, 0) + 1
            labels = []
            scores = []
            for answer in answer_count:
                assert answer in ans2label
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)
            assert q['answer_type'].strip().lower() == "closed" or q['answer_type'].strip().lower() == "open"
            answer_type = 0 if q['answer_type'].strip().lower() == "closed" else 1
            _annotation[q["img_path"]][q["qid"]].append(
                {"labels": labels, "scores": scores, "answer_type": answer_type})

    # Write to the files
    for split in ["test"]:
        annot = annotations[split]
        annot_paths = [path for path in annot if os.path.exists(path)]
        print("######", len(annot_paths), len(annot))
        assert len(annot_paths) == len(annot) or len(annot_paths) == len(annot) - 1
        print("{} set: {} images, {} questions".format(split,
                                                       len(annot),
                                                       len([vv for k, v in annot.items() for kk, vv in v.items()])))

        bs = [
            path2rest_vqa(path, split, annotations, label2ans) for path in tqdm(annot_paths)
        ]
        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "answer_type",
                "split",
            ],
        )

        print("#########max", max(dataframe["answer_labels"]))


        table = pa.Table.from_pandas(dataframe)

        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_melinda(path, iid2captions, iid2i_meth, iid2p_meth, iid2i_meth_label, iid2p_meth_label, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    i_meth = iid2i_meth[name]
    p_meth = iid2p_meth[name]
    i_meth_label = iid2i_meth_label[name]
    p_meth_label = iid2p_meth_label[name]
    assert len(captions) == len(i_meth)
    assert len(captions) == len(p_meth)
    assert len(captions) == len(i_meth_label)
    assert len(captions) == len(p_meth_label)
    split = iid2split[name]
    return [binary, captions, name, i_meth, p_meth, i_meth_label, p_meth_label, split]


def make_arrow_melinda(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2i_meth = defaultdict(list)
    iid2p_meth = defaultdict(list)
    iid2i_meth_label = defaultdict(list)
    iid2p_meth_label = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2split[sample["img_path"]] = split
            iid2i_meth[sample["img_path"]].append(sample["i_meth"])
            iid2p_meth[sample["img_path"]].append(sample["p_meth"])
            iid2i_meth_label[sample["img_path"]].append(sample["i_meth_label"])
            iid2p_meth_label[sample["img_path"]].append(sample["p_meth_label"])

    i_meth_set = set([vv for k, v in iid2i_meth.items() for vv in v])
    i_meth_label_set = set([vv for k, v in iid2i_meth_label.items() for vv in v])
    p_meth_set = set([vv for k, v in iid2p_meth.items() for vv in v])
    p_meth_label_set = set([vv for k, v in iid2p_meth_label.items() for vv in v])

    i_meth_set = sorted(i_meth_set)
    i_meth_label_set = sorted(i_meth_label_set)
    p_meth_set = sorted(p_meth_set)
    p_meth_label_set = sorted(p_meth_label_set)

    i_meth_dict = {j: i for i, j in enumerate(i_meth_set)}
    p_meth_dict = {j: i for i, j in enumerate(p_meth_set)}
    i_meth_label_dict = {j: i for i, j in enumerate(i_meth_label_set)}
    p_meth_label_dict = {j: i for i, j in enumerate(p_meth_label_set)}

    iid2i_meth = {k: [i_meth_dict[vv] for vv in v] for k, v in iid2i_meth.items()}
    iid2p_meth = {k: [p_meth_dict[vv] for vv in v] for k, v in iid2p_meth.items()}
    iid2i_meth_label = {k: [i_meth_label_dict[vv] for vv in v] for k, v in iid2i_meth_label.items()}
    iid2p_meth_label = {k: [p_meth_label_dict[vv] for vv in v] for k, v in iid2p_meth_label.items()}

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    bs = [path2rest_melinda(path, iid2captions, iid2i_meth, iid2p_meth, iid2i_meth_label, iid2p_meth_label, iid2split)
          for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "i_meth", "p_meth", "i_meth_label",
                                                   "p_meth_label", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_chexpert(path, iid2captions, iid2chexpert, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    chexpert = iid2chexpert[name]
    split = iid2split[name]
    return [binary, captions, name, chexpert, split]


def make_arrow_chexpert(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    for split, split_data in data.items():
        iid2captions = defaultdict(list)
        iid2chexpert = defaultdict(list)
        iid2split = dict()

        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2chexpert[sample["img_path"]].extend(sample["chexpert"])
            iid2split[sample["img_path"]] = split

        path = len(iid2captions)
        caption_paths = [path for path in iid2captions if os.path.exists(path)]
        print(f"+ {len(caption_paths)} images / {path} annotations")
        bs = [path2rest_chexpert(path, iid2captions, iid2chexpert, iid2split) for path in tqdm(caption_paths)]

        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "chexpert", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_pnsa_pneumonia(path, iid2captions, iid2pnsa_pneumonia, iid2split):
    name = path

    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    pnsa_pneumonia = iid2pnsa_pneumonia[name]
    split = iid2split[name]
    return [binary, captions, name, pnsa_pneumonia, split]


def make_arrow_pnsa_pneumonia(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    for split, split_data in data.items():
        iid2captions = defaultdict(list)
        iid2pnsa_pneumonia = defaultdict(list)
        iid2split = dict()

        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2pnsa_pneumonia[sample["img_path"]].extend(sample["pnsa_pneumonia"])
            iid2split[sample["img_path"]] = split

        path = len(iid2captions)
        caption_paths = [path for path in iid2captions if os.path.exists(path)]
        print(f"+ {len(caption_paths)} images / {path} annotations")
        bs = [path2rest_pnsa_pneumonia(path, iid2captions, iid2pnsa_pneumonia, iid2split) for path in
              tqdm(caption_paths)]

        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "pnsa_pneumonia", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_clm_mimic_cxr(path, iid2captions, iid2findings, iid2impression, iid2chexpert, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    findings = iid2findings[name]
    impression = iid2impression[name]
    assert len(captions) == 1 and len(impression) == 1
    chexpert = iid2chexpert[name]
    split = iid2split[name]
    return [binary, captions, name, findings, impression, chexpert, split]


def make_arrow_clm_mimic_cxr(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2findings = defaultdict(list)
    iid2impression = defaultdict(list)
    iid2chexpert = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2findings[sample["img_path"]].extend(sample["findings"])
            iid2impression[sample["img_path"]].extend(sample["impression"])
            iid2chexpert[sample["img_path"]].extend(sample["chexpert"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    bs = [path2rest_clm_mimic_cxr(path, iid2captions, iid2findings, iid2impression, iid2chexpert, iid2split) for path in
          tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "findings", "impression",
                                                   "chexpert", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_text_classification(guid, iid2text_a, iid2labels, iid2split):
    text_a = iid2text_a[guid]
    labels = iid2labels[guid]
    split = iid2split[guid]
    assert len(text_a) == 1
    return [text_a, guid, labels, split]


def make_arrow_text_classification(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2text_a = defaultdict(list)
    iid2labels = dict()
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2text_a[sample["guid"]].extend(sample["text_a"])
            iid2labels[sample["guid"]] = sample["label"]
            iid2split[sample["guid"]] = split

    statistics(iid2text_a, iid2split)
    bs = [path2rest_text_classification(guid, iid2text_a, iid2labels, iid2split) for guid in tqdm(iid2text_a)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["text_a", "guid", "label", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_nli(guid, iid2text_a, iid2text_b, iid2text, iid2labels, iid2split):
    text_a = iid2text_a[guid]
    text_b = iid2text_b[guid]
    text = iid2text[guid]
    labels = iid2labels[guid]
    split = iid2split[guid]
    assert len(text_a) == 1
    assert len(text_b) == 1
    assert len(text) == 1
    return [text_a, text_b, text, guid, labels, split]


def make_arrow_text_nli(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2text_a = defaultdict(list)
    iid2text_b = defaultdict(list)
    iid2text = defaultdict(list)
    iid2labels = dict()
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2text_a[sample["guid"]].extend(sample["text_a"])
            iid2text_b[sample["guid"]].extend(sample["text_b"])
            iid2text[sample["guid"]].extend(sample["text"])
            iid2labels[sample["guid"]] = sample["label"]
            iid2split[sample["guid"]] = split

    statistics(iid2text_a, iid2split)
    statistics(iid2text_b, iid2split)
    statistics(iid2text, iid2split)
    bs = [path2rest_nli(guid, iid2text_a, iid2text_b, iid2text, iid2labels, iid2split) for guid in tqdm(iid2text_a)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["text_a", "text_b", "text", "guid", "label", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_word(token):
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
                re.search(comma_strip, token) != None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token

def prepro_robot_demo():
    random.seed(42)

    data = {
        "test": []
    }
    user_data_path = '/home/coder/projects/SystemDataset/robot/robot.csv'
    final_data_df = pd.read_csv(user_data_path)
    img_path = str(final_data_df['img_path'][0])
    content = str(final_data_df['content'][0])

    # for split in ["train", "val", "test"]:
    data["test"].append({
        "img_path": img_path,
        "qid": 0,
        "question": content,
        "answer": 'Yes',
        "answer_type": 'CLOSED'
    })
    make_arrow_vqa(data, "vqa", "/home/coder/projects/METER/data/vqa_robot_demo")


if __name__ == '__main__':
    prepro_robot_demo()
