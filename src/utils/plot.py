import os
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting utility")
    parser.add_argument("--ideal_path", type=str, help="Input file")
    parser.add_argument("--sleb_path", type=str, help="Input file")
    parser.add_argument("--gene_path", type=str, help="Input file")
    parser.add_argument("--output_path", type=str, help="Output file")
    parser.add_argument("--ub", type=float, help="Upper bound", default=None)
    parser.add_argument("--lb", type=float, help="Lower bound", default=None)
    args = parser.parse_args()
    model_name = args.ideal_path.split("/")[-1].split("_")[-1].strip(".json")

    with open(args.ideal_path, "r") as f:
        sparsity_loss = json.load(f)
        sparsity_loss = {int(k): v for k, v in sparsity_loss.items()}
        if args.ub is not None:
            sparsity_loss = {k: v for k, v in sparsity_loss.items() if k <= args.ub}
        if args.lb is not None:
            sparsity_loss = {k: v for k, v in sparsity_loss.items() if k >= args.lb}
    
    if args.sleb_path is not None:
        sleb_sparsity_loss = {}
        with open(args.sleb_path, "r") as f:
            lines = f.readlines()
        sparsity = None
        loss = None
        for line in lines:
            if line.startswith("# Total Blocks:"):
                sparsity = float(line.split(":")[1].strip())
            if line.startswith("# Remove Blocks:"):
                sparsity = float(line.split(":")[1].strip()) / sparsity
                sparsity = int(round(sparsity, 2) * 100)
            if line.startswith("Final Loss:"):
                loss = float(line.split(":")[1].strip())
                sleb_sparsity_loss[sparsity] = loss
        if args.ub is not None:
            sleb_sparsity_loss = {k: v for k, v in sleb_sparsity_loss.items() if k <= args.ub}
        if args.lb is not None:
            sleb_sparsity_loss = {k: v for k, v in sleb_sparsity_loss.items() if k >= args.lb}
        print(f"SLEB Loss")
        for sparsity, loss in sleb_sparsity_loss.items():
            print(f"{sparsity}", end="\t")
        print("\n")
        for sparsity, loss in sleb_sparsity_loss.items():
            print(f"{loss}", end="\t")
        print("\n")
    
    if args.gene_path is not None:
        gene_sparsity_loss = {}
        with open(args.gene_path, "r") as f:
            lines = f.readlines()
        sparsity = None
        loss = None
        for line in lines:
            if "Max reduced layers:" in line:
                sparsity = float(line.split(":")[-1].strip()) / 22
                sparsity = int(round(sparsity, 2) * 100)
            if line.startswith("Objective function:"):
                loss = float(line.split(":")[1].strip())
                gene_sparsity_loss[sparsity] = loss
        if args.ub is not None:
            gene_sparsity_loss = {k: v for k, v in gene_sparsity_loss.items() if k <= args.ub}
        if args.lb is not None:
            gene_sparsity_loss = {k: v for k, v in gene_sparsity_loss.items() if k >= args.lb}
        print(f"Our Loss")
        for sparsity, loss in gene_sparsity_loss.items():
            print(f"{sparsity}", end="\t")
        print("\n")
        for sparsity, loss in gene_sparsity_loss.items():
            print(f"{loss}", end="\t")
        print("\n")


    fig = plt.figure()
    fig_data = [(sparsity, loss) for sparsity, loss in sorted(sparsity_loss.items(), key=lambda x: x[0])]
    sparsities, losses = zip(*fig_data)
    if sparsities[0] == 0:
        sparsities = sparsities[1:]
        origin_loss = losses[0]
        if len(origin_loss) > 1:
            origin_loss = origin_loss[0]
        losses = losses[1:]
        print(f"Origin Loss: {origin_loss}")
    else:
        origin_loss = None
    print("Ideal Loss")
    for sparsity, loss in zip(sparsities, losses):
        print(f"{sparsity}", end="\t")
    print("\n")
    for sparsity, loss in zip(sparsities, losses):
        print(f"{np.min(loss)}", end="\t")
    print("\n")
    # plt.boxplot(losses, labels=[f"{s}%" for s in sparsities])
    if origin_loss is not None:
        plt.axhline(y=origin_loss, color='r', linestyle='--', label='Original Loss')
    plt.plot(sleb_sparsity_loss.keys(), sleb_sparsity_loss.values(), 'b--', label='SLEB Loss')
    plt.plot(sparsities, [np.min(losses[i]) for i in range(len(sparsities))], 'g--', label='Ideal Loss')
    plt.plot(gene_sparsity_loss.keys(), gene_sparsity_loss.values(), 'p--', label='Gene Loss')
    plt.xlim(args.lb, args.ub)
    plt.xlabel("Sparsity")
    plt.ylabel("Loss")
    # plt.title("Loss Distribution by Sparsity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(args.output_path)
    plt.close()
    