# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import torch
import unittest
import bittensor as bt

from neurons.validator import Neuron as Validator
from neurons.miner import Neuron as Miner

from text_recognition.protocol import TextRecognitionSynapse
from text_recognition.utils.image_processing import load_image
from text_recognition.validator.forward import forward
from text_recognition.utils.uids import get_random_uids
from text_recognition.validator.reward import get_rewards
from text_recognition.base.validator import BaseValidatorNeuron


class ValidatorNeuronTestCase(unittest.TestCase):
    """
    This class contains unit tests for the RewardEvent classes.

    The tests cover different scenarios where completions may or may not be successful and the reward events are checked that they don't contain missing values.
    The `reward` attribute of all RewardEvents is expected to be a float, and the `is_filter_model` attribute is expected to be a boolean.
    """

    def setUp(self):
        sys.argv = sys.argv[0] + ["--config", "tests/configs/validator.json"]

        config = BaseValidatorNeuron.config()
        config.wallet._mock = True
        config.metagraph._mock = True
        config.subtensor._mock = True
        self.neuron = Validator(config)
        self.miner_uids = get_random_uids(self, k=10)

    def test_run_single_step(self):
        pass

    def test_sync_error_if_not_registered(self):
        pass

    def test_forward(self):

        responses = self.neuron.dendrite.query(
            axons=[
                self.neuron.metagraph.axons[uid] for uid in self.miner_uids
            ],
            synapse=TextRecognitionSynapse(image_input=load_image("astronaut.jpg")),
            deserialize=False,
        )

        for i, response in enumerate(responses):
            self.assertEqual(response, "ASTRONAUT")

    def test_reward(self):
        responses = self.dendrite.query(
            # Send the query to miners in the network.
            axons=[self.metagraph.axons[uid] for uid in self.miner_uids],
            synapse=TextRecognitionSynapse(image_input=load_image("astronaut.jpg")),
            deserialize=False,
        )

        rewards = get_rewards(self.neuron, responses)
        expected_rewards = torch.FloatTensor([1.0] * len(responses))
        self.assertEqual(rewards, expected_rewards)

    def test_reward_with_nan(self):
        """
        Test that NaN rewards are correctly sanitized and a warning is logged.
        """
        responses = [
            TextRecognitionSynapse(text_recognition_output="ASTRONAUT")
        ] * len(self.miner_uids)
        rewards = get_rewards(self.validator, responses)

        rewards[0] = float("nan")

        with self.assertLogs("bittensor", level="WARNING") as cm:
            sanitized_rewards = self.validator.sanitize_rewards(rewards)
            self.assertNotIn(
                float("nan"), sanitized_rewards, "NaN reward was not sanitized"
            )
            self.assertTrue(
                any("WARNING" in message for message in cm.output),
                "Expected warning log for NaN reward",
            )
