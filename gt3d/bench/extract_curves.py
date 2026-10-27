"""Extract accuracy-vs-time curves from train_dist logs: one CSV row per evaluated epoch
(epoch, cumulative train seconds, train/val/test accuracy, best-so-far test at best val).

    python bench/extract_curves.py logs/products_accuracy_h512.log logs/reddit_acc_h128.log > curves.csv
"""
import re
import sys

PAT = re.compile(r"^epoch\s+(\d+)\s+loss\s+([\d.]+)\s+train\s+([\d.]+)\s+val\s+([\d.]+)\s+test\s+([\d.]+)\s+\|\s+best val\s+([\d.]+)\s+test\s+([\d.]+)\s+@\s+(\d+)\s+\|\s+([\d.]+)\s+ms/epoch")


def main(paths):
    print("log,epoch,train_seconds,loss,train_acc,val_acc,test_acc,best_val,test_at_best_val,best_epoch")
    for p in paths:
        for line in open(p):
            m = PAT.match(line)
            if not m:
                continue
            ep, loss, tr, va, te, bv, bt, be, ms = m.groups()
            secs = int(ep) * float(ms) / 1000.0            # ms/epoch printed as the running mean over epochs so far
            print(f"{p},{ep},{secs:.1f},{loss},{tr},{va},{te},{bv},{bt},{be}")


if __name__ == "__main__":
    main(sys.argv[1:])
