def calculate_f1(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Example usage
precision = 0.73
recall = 0.27
f1_score = calculate_f1(precision, recall)
print(f"F1-score: {f1_score:.3f}")
