import torch
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


def soft_crossentropy(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


class CustomBertForSequenceSimilarity(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.feature_extractor = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = torch.nn.Linear(config.hidden_size * 4, 1)

        self.label = None
        # if config.choice_size is not None:
        #     self.label = torch.tensor([1.] + [0.] * (config.choice_size - 1))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        doctor_input_ids=None,
        doctor_attention_mask=None,
        doctor_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        patient_turn = input_ids.size(1) # Expected to be 2
        patient_sample = input_ids.size(0)
        doctor_choice = doctor_input_ids.size(1) # Default 10 choices, first choice is correct
        doctor_sample = doctor_input_ids.size(0)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        doctor_input_ids = doctor_input_ids.view(-1, doctor_input_ids.size(-1))
        doctor_attention_mask = doctor_attention_mask.view(-1, doctor_attention_mask.size(-1))
        doctor_token_type_ids = doctor_token_type_ids.view(-1, doctor_token_type_ids.size(-1))

        patient_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        doctor_outputs = self.bert(
            doctor_input_ids,
            attention_mask=doctor_attention_mask,
            token_type_ids=doctor_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        patient_outputs = patient_outputs[1].view(patient_sample, -1)
        patient_outputs = self.feature_extractor(patient_outputs)
        doctor_outputs = doctor_outputs[1].view(doctor_sample, doctor_choice, -1)

        # Stack with same size as doctor choice
        patient_outputs = torch.stack([patient_outputs] * doctor_choice, dim=1)
        # print(patient_outputs.shape)
        # print(doctor_outputs.shape)
        
        # [a, b, a - b, a * b] 
        merged_output = torch.cat([patient_outputs, doctor_outputs, patient_outputs - doctor_outputs, patient_outputs * doctor_outputs], dim=-1)
        
        logits = self.classifier(merged_output).view(-1)

        labels = torch.tensor(([1.] + [0.] * (doctor_choice - 1)) * patient_sample).to(self.device)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )