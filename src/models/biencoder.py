import torch
import torch.nn as nn
from transformers import BertModel

class BiEncoderClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=3, dropout=0.1):
        super(BiEncoderClassifier, self).__init__()

        # Shared BERT encoder — both resume and JD pass through this
        self.bert = BertModel.from_pretrained(model_name)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Classifier head
        # Input: 768*4=3072 (concat + diff + product)
        # Output: num_classes (3)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 4, num_classes)

    def encode(self, input_ids, attention_mask):
        """
        Pass tokens through BERT and return the [CLS] token embedding.
        [CLS] is always at position 0 and represents the whole sequence.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # last_hidden_state shape: (batch_size, seq_len, 768)
        # We take position 0 = [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward(self, resume_input_ids, resume_attention_mask,
                jd_input_ids, jd_attention_mask):
        # Encode resume and JD separately through the same BERT
        resume_emb = self.encode(resume_input_ids, resume_attention_mask)
        jd_emb     = self.encode(jd_input_ids, jd_attention_mask)

        # Combine embeddings — same strategy as Baseline 2
        combined = torch.cat([
            resume_emb,
            jd_emb,
            torch.abs(resume_emb - jd_emb),
            resume_emb * jd_emb
        ], dim=1)  # shape: (batch_size, 3072)

        combined = self.dropout(combined)
        logits   = self.classifier(combined)  # shape: (batch_size, 3)

        return logits