import torch
from fastchat import model

class autodan_SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        self.conv_template.messages = []

        self.conv_template.append_message(self.conv_template.roles[0], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._user_role_slice = slice(None, len(toks))

        self.conv_template.update_last_message(f"{self.adv_string}")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
        self._control_slice = self._goal_slice

        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

        self.conv_template.update_last_message(f"{self.target}")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2)
        self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 3)


        self.conv_template.messages = []

        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids