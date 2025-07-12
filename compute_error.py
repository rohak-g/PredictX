# compute_error.py

def calculate_average_error(txt_file):
    predicted = []
    actual = []

    with open(txt_file, "r") as f:
        lines = f.readlines()[2:]  # Skip the first two header lines

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    pred = float(parts[2])
                    act = float(parts[4])
                    predicted.append(pred)
                    actual.append(act)
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")

    if not predicted:
        print("No valid data found.")
        return

    # Calculate errors
    abs_errors = [abs(p - a) for p, a in zip(predicted, actual)]
    pct_errors = [abs(p - a) / a * 100 for p, a in zip(predicted, actual) if a != 0]

    print(f"Average Absolute Error: {sum(abs_errors) / len(abs_errors):.4f}")
    print(f"Average Percentage Error: {sum(pct_errors) / len(pct_errors):.2f}%")


if __name__ == "__main__":
    txt_path = "test_predictions.txt"  # Change path if needed
    calculate_average_error(txt_path)
