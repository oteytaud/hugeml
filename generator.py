# Copyright 2007 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import hashlib

def gen_labelling_function(label_generator_name):
  """Creates the ground truth function.

  Args:
    label_generator_name: selected ground truth generator; can be: xor,
      majority, parity_onemax, parity_leadingones, needle, rote, smooth4_parity,
      smooth8_parity, smooth4_parity_leadingones or smooth8_parity_leadingones.

  Returns:
    A lambda expression taking as input a list of critical feature values (
    before pre-processing), and returning the ground truth (also a boolean
    value).
  """

  if label_generator_name == "xor":
    return lambda x: sum(x) % 2
  elif label_generator_name == "majority":
    return lambda x: 1 if sum(x) >= len(x) // 2 else 0
  elif label_generator_name == "parity_onemax":
    return lambda x: sum(x) % 2
  elif label_generator_name == "parity_leadingones":
    return lambda x: sum(np.cumprod(x)) % 2
  elif label_generator_name == "needle":
    return lambda x: np.prod(x)
  elif label_generator_name == "rote":
    return lambda x: int(hashlib.sha224(str([x])).hexdigest(), 16) % 2
  elif label_generator_name == "smooth4_parity":
    return lambda x: (sum(x) // 4) % 2
  elif label_generator_name == "smooth8_parity":
    return lambda x: (sum(x) // 8) % 2
  elif label_generator_name == "smooth4_parity_leadingones":
    return lambda x: (sum(np.cumprod(x)) // 4) % 2
  elif label_generator_name == "smooth8_parity_leadingones":
    return lambda x: (sum(np.cumprod(x)) // 8) % 2


def gen_critical_feature_values(num_critical_features):
  """Creates a vector of value for the critical features (before flip/shuffle).

  Args:
    - num_critical_features: number of critical features.

  Returns:
    A list of boolean value of length num_critical_features.
  """
  return [np.random.choice([0, 1]) for _ in range(num_critical_features)]


def gen_all_features_generator(num_critical_features, num_useless_features):
  """Creates a function that returns a vector of input feature values.

  Args:
    - num_useless_features: number of useless features.

  Returns:
    A function taking as input the critical feature values (i.e. the result of
    "gen_critical_feature_values"), and returns a list of boolean values of
    length num_critical_features.
  """

  # Which feature to flip.
  flip_critical_feature = np.random.randint(
      2, size=num_critical_features + num_useless_features)
  # The new order of the features.
  shuffling_order = np.arange(num_critical_features + num_useless_features)
  np.random.shuffle(shuffling_order)

  def all_features_generator(critical_feature_values):
    useless_feature_values = [
        np.random.choice([0, 1]) for _ in range(num_useless_features)
    ]
    all_features = critical_feature_values + useless_feature_values

    # Flip feature values.
    all_features = [(value + sym) % 2
                    for (value,
                         sym) in zip(all_features, flip_critical_feature)]
    # Shuffle features.
    all_features = [all_features[idx] for idx in shuffling_order]
    return all_features

  return all_features_generator


def generate_dataset(label_generator_name, num_critical_features,
                     num_useless_features, num_examples):
  """Returns a list of examples.

  Args:
    - label_generator_name: selected ground truth generator. See
      "gen_labelling_function" for the possible values.
    - num_critical_features: number of critical features.
    - num_useless_features: number of useless features.
    - num_examples: number of examples to generate.

  Returns:
    A list of boolean value of length num_critical_features. Each example is a
    tuple (input features, label).
  """

  # Check the random generator. Should be fine for Python version >= 2.3.
  np.random.seed(7)
  assert int(np.random.uniform() * 10000000) == 763082
  assert int(np.random.uniform() * 10000000) == 7799187

  labelling = gen_labelling_function(label_generator_name)
  all_features_generator = gen_all_features_generator(num_critical_features,
                                                      num_useless_features)
  examples = []
  for example_index in xrange(num_examples):
    np.random.seed(example_index % (2**32))
    critical_feature_values = gen_critical_feature_values(num_critical_features)
    label_value = labelling(critical_feature_values)
    np.random.seed((8*example_index) % 10619863)
    all_feature_values = all_features_generator(critical_feature_values)
    examples.append((all_feature_values, label_value))

    # Check one of the example
    if (label_generator_name == "xor" and example_index == 2 and
        num_useless_features == 11 and num_critical_features == 11):
      assert all_feature_values == [
          0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1
      ], str(all_feature_values)
      assert label_value == 1

  return examples

for record in generate_dataset("xor", 11, 11, 33):
  print record

import csv

candidate_num_critical_attributes = [11, 14, 17, 18, 19, 20, 21]
candidate_labels = ["xor", "majority", "parity_onemax", "parity_leadingones",
           "needle", "rote", "smooth4_parity", "smooth8_parity",
           "smooth4_parity_leadingones", "smooth8_parity_leadingones"]
for label in candidate_labels:
  for num_critical_attributes in candidate_num_critical_attributes:
    for num_useless_attributes in [0, num_critical_attributes]:
      # Real (very slow, to be distributed) case, 3^num_critical_attributes
      # examples:
      # n = 3**num_critical_attributes
      # More realistic, on a single computer.
      n = 100

      data = generate_dataset(label, num_critical_attributes,
          num_useless_attributes, n)
      with open(label + "_dim" + str(num_critical_attributes) + "_" + str(
          num_useless_attributes) + "uselessvars" + "_" + str(n) + ".csv", "w") as f:
        csv.writer(f).writerows(data)
