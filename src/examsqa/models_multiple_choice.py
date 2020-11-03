import torch
from transformers import BertForMaskedLM, RobertaForMaskedLM, XLMRobertaForMaskedLM


class BertLMForMultipleChoice(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_positions=None,
        masked_input_ids=None,
        labels=None,
    ):
        batch_size, num_choices, num_words = input_ids.shape

        input_ids = input_ids.view(-1, input_ids.size(-1))
        masked_input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        )
        position_ids = (
            position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        )

        prediction_scores = super().forward(
            masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        predicted_probs = torch.softmax(
            prediction_scores[0].view(batch_size, num_choices, num_words, -1), dim=-1
        ).view(-1, self.config.vocab_size)

        log_probs = torch.log(
            predicted_probs.gather(-1, input_ids.view(-1, 1)).view(
                batch_size, num_choices, num_words
            )
        ).masked_fill(masked_positions == 0, 0)

        scores = 1 / (-log_probs.sum(dim=-1) / masked_positions.sum(axis=-1))

        return torch.FloatTensor(0), scores


class RobertaLMForMultipleChoice(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_positions=None,
        masked_input_ids=None,
        labels=None,
    ):
        batch_size, num_choices, num_words = input_ids.shape

        input_ids = input_ids.view(-1, input_ids.size(-1))
        masked_input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        )
        position_ids = (
            position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        )

        prediction_scores = super().forward(
            masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        predicted_probs = torch.softmax(
            prediction_scores[0].view(batch_size, num_choices, num_words, -1), dim=-1
        ).view(-1, self.config.vocab_size)

        log_probs = torch.log(
            predicted_probs.gather(-1, input_ids.view(-1, 1)).view(
                batch_size, num_choices, num_words
            )
        ).masked_fill(masked_positions == 0, 0)

        scores = 1 / (-log_probs.sum(dim=-1) / masked_positions.sum(axis=-1))

        return torch.FloatTensor(0), scores


class XLMRobertaLMForMultipleChoice(XLMRobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_positions=None,
        masked_input_ids=None,
        labels=None,
    ):
        batch_size, num_choices, num_words = input_ids.shape

        input_ids = input_ids.view(-1, input_ids.size(-1))
        masked_input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = (
            attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        )
        position_ids = (
            position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        )

        prediction_scores = super().forward(
            masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        predicted_probs = torch.softmax(
            prediction_scores[0].view(batch_size, num_choices, num_words, -1), dim=-1
        ).view(-1, self.config.vocab_size)

        log_probs = torch.log(
            predicted_probs.gather(-1, input_ids.view(-1, 1)).view(
                batch_size, num_choices, num_words
            )
        ).masked_fill(masked_positions == 0, 0)

        scores = 1 / (-log_probs.sum(dim=-1) / masked_positions.sum(axis=-1))

        return torch.FloatTensor(0), scores
