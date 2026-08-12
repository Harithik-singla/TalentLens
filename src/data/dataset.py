import torch
from torch.utils.data import Dataset

class ResumeJDDataset(Dataset):
    def __init__(self, encodings_path):
        """
        encodings_path : path to a .pt file saved in Week 1
                         contains dict of tensors: resume_input_ids,
                         resume_attention_mask, jd_input_ids,
                         jd_attention_mask, labels
        """
        data = torch.load(encodings_path)

        self.resume_input_ids      = data['resume_input_ids']
        self.resume_attention_mask = data['resume_attention_mask']
        self.jd_input_ids          = data['jd_input_ids']
        self.jd_attention_mask     = data['jd_attention_mask']
        self.labels                = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'resume_input_ids'      : self.resume_input_ids[idx],
            'resume_attention_mask' : self.resume_attention_mask[idx],
            'jd_input_ids'          : self.jd_input_ids[idx],
            'jd_attention_mask'     : self.jd_attention_mask[idx],
            'label'                 : self.labels[idx]
        }