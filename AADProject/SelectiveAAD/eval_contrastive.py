"""
1. Load trained model.
2. Build test SelectiveAADDataset with full trials or sliding windows.
3. Evaluate:
    window-level accuracy: fraction of windows where argmax similarity is the attended speaker.
    trial-level accuracy: majority vote over windows in a trial.

"""

def evaluate_model(model, dataloader):
    model.eval()
    all_correct = 0
    all_total = 0
    # optionally aggregate per trial