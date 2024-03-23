from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqa_train", "vqa_val"]
            ## 测试代码，记得删掉
            # names = ["vqa_val"]
        elif split == "val":
            names = ["vqa_val"]
        elif split == "test":
            names = ["vqa_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        ## later by hxj 
        # if self.split != "test":
        #     answers = self.table["answers"][index][question_index].as_py()
        #     labels = self.table["answer_labels"][index][question_index].as_py()
        #     scores = self.table["answer_scores"][index][question_index].as_py()
        #     answer_types = self.table["answer_type"][index][question_index].as_py()
        # else:
        #     answers = list()
        #     labels = list()
        #     scores = list()

        answers = self.table["answers"][index][question_index].as_py()
        labels = self.table["answer_labels"][index][question_index].as_py()
        scores = self.table["answer_scores"][index][question_index].as_py()
        answer_types = self.table["answer_type"][index][question_index].as_py()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "answer_types": answer_types,
            "qid": qid,
        }
