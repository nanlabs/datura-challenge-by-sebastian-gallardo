# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Sebastian Gallardo

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

import random
import bittensor as bt

from template.protocol import TextRecognitionSynapse
from template.utils import image_processing
from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids


def get_sample():
    '''
    Returns a random sample from the list of samples.
    This could be improved by generating a random image and adding noise to increase difficulty.
    '''
    samples = [
        {
            "filename": "astronaut.jpg",
            "expected_text": "ASTRONAUT"
        },
        {
            "filename": "memory.jpg",
            "expected_text": "MEMORY"
        },
    ]
    return random.choice(samples)

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    sample = get_sample()
    image_input = image_processing.load_image(sample["filename"])

    # get_random_uids is an example method, but you can replace it with your own.
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    # The dendrite client queries the network.
    responses = self.dendrite.query(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=TextRecognitionSynapse(image_input),
        deserialize=False,
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    rewards = get_rewards(self, expected_text=sample["expected_text"], responses=responses)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)