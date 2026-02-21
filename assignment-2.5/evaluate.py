import json
import math
import sys

if len(sys.argv) != 3:
    print("Usage: python evaluate.py path/to/example-test-results.json path/to/your-results.json")
    sys.exit(1)

with open(sys.argv[1], "r") as f:
    example_results = json.load(f)

with open(sys.argv[2], "r") as f:
    your_results = json.load(f)

if len(example_results) != len(your_results):
    print("Number of test results do not match")
    sys.exit(1)

correct = 0
for i in example_results:
    if example_results[i] == your_results[i]:
        correct += 1

print(f"Accuracy: {correct / len(example_results)}")
points = math.ceil(215 * (correct / len(example_results) - 1/4))
print(f"If this were the test data, you would receive {points} points")

