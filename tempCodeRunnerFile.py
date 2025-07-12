
            correct += ((pred >= lower) & (pred <= upper)).sum().item()
            total += batch.num_graphs
    print(f"Final Accuracy (±5%): {100 * correct / total:.2f}%")


def train():
    root = r"C:/Users/gupta/Desktop/vlsi/delay_predictor/Dataset/ordered/data1"
    dirs = [d for d in os.listdir(root) if os.pat