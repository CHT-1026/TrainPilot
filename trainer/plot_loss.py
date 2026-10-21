"""读取训练 stdout 日志或 metrics.jsonl，绘制损失和学习率。

python -m trainer.plot_loss --log runs/qwen-fullsft/train.log
累计均值不等于区间均值；旧日志缺少 token 计数时不反推区间损失。
"""
import argparse
from dataclasses import dataclass, field
import importlib.util
import json
import math
from pathlib import Path
import re


class LossWindow:
    """只在日志输出时推进，计算相邻日志区间的有效 token 加权平均。"""
    def __init__(self, state=None):
        self.loss_sum = state["loss_sum"] if state else 0.0
        self.valid_tokens = state["valid_tokens"] if state else 0

    def update(self, state):
        count = state["valid_tokens"] - self.valid_tokens
        if count <= 0:
            raise ValueError("日志区间必须包含新的有效监督 token")
        value = (state["loss_sum"] - self.loss_sum) / count
        self.loss_sum, self.valid_tokens = state["loss_sum"], state["valid_tokens"]
        return value


@dataclass
class LossHistory:
    updates: list = field(default_factory=list)
    epochs: list = field(default_factory=list)
    validations: list = field(default_factory=list)
    notes: list = field(default_factory=list)


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FIELDS = re.compile(rf"\b([A-Za-z_]+)=({NUMBER})(?=\s|$|[;,；])")


def read_history(path):
    path = Path(path)
    history = LossHistory()
    end_steps, accumulation = {}, None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if path.suffix.lower() == ".jsonl":
            row = json.loads(line) if line.strip() else {}
            event = row.get("event")
            if event == "update":
                history.updates.append({"epoch":row["epoch"], "step":row["global_step"],
                    "loss":row["mean_loss"], "window_loss":row.get("window_loss"),
                    "lr":row.get("learning_rate", row.get("lr"))})
            elif event == "epoch":
                history.epochs.append({"epoch":row["completed_epochs"], "step":row["global_step"],
                    "train_loss":row["train"]["loss"], "val_loss":row["validation"]["loss"]})
            elif event == "initial_validation":
                value = row.get("validation", row).get("loss")
                if value is not None:
                    history.validations.append({"step":row.get("global_step", 0), "loss":value})
            elif event == "run_end":
                end_steps[row["completed_epochs"]] = row["global_step"]
            continue
        values = {name:float(value) for name, value in FIELDS.findall(line)}
        if "accumulation" in values:
            accumulation = int(values["accumulation"])
        chinese_accumulation = re.search(r"梯度累积=(\d+)", line)
        if chinese_accumulation:
            accumulation = int(chinese_accumulation.group(1))
        epoch = re.search(r"\bepoch=(\d+)(?:/\d+)?", line)
        batch = re.search(r"\bbatch=(\d+)/(\d+)", line)
        step = int(values["global_step"]) if "global_step" in values else None
        if "completed_epochs" in values and step is not None:
            end_steps[int(values["completed_epochs"])] = step
        if "训练前验证" in line and "loss" in values:
            history.validations.append({"step":0, "loss":values["loss"]})
        if epoch and "train_loss" in values:
            history.epochs.append({"epoch":int(epoch.group(1)), "step":step,
                "train_loss":values["train_loss"], "val_loss":values.get("val_loss")})
        elif epoch and batch and step is not None:
            loss = values.get("mean_loss", values.get("loss"))
            if loss is not None:
                history.updates.append({"epoch":int(epoch.group(1)), "step":step,
                    "loss":loss, "window_loss":values.get("window_loss"), "lr":values.get("lr"),
                    "batch":int(batch.group(1)), "batches":int(batch.group(2))})
    # 同一实验恢复到旧断点后可能重新记录某些 step，以最后一次记录为准。
    unique = {(row["epoch"], row["step"]):row for row in history.updates}
    if len(unique) != len(history.updates):
        history.notes.append("重复的 epoch/step 使用最后记录；不要拼接不同实验的日志。")
    history.updates = sorted(unique.values(), key=lambda row:row["step"])
    history.epochs = list({row["epoch"]:row for row in history.epochs}.values())
    for row in history.epochs:
        if row["step"] is None:
            row["step"] = end_steps.get(row["epoch"])
            previous = [x for x in history.updates if x["epoch"] == row["epoch"]]
            if row["step"] is None and previous and accumulation:
                last = previous[-1]
                row["step"] = last["step"] + math.ceil((last["batches"]-last["batch"])/accumulation)
        if row["step"] is not None and row["val_loss"] is not None:
            history.validations.append({"step":row["step"], "loss":row["val_loss"]})
    history.validations = sorted({row["step"]:row for row in history.validations}.values(), key=lambda row:row["step"])
    if not history.updates and not history.validations:
        raise ValueError(f"未在 {path} 中找到损失记录")
    return history


def plot_loss(log_path, output=None, title=None):
    # 绘图是训练结束后的附加产物，不因旧环境缺少绘图库影响已保存的模型。
    if importlib.util.find_spec("matplotlib") is None:
        print("未生成损失图：请安装 matplotlib，再运行 python -m trainer.plot_loss --log", log_path)
        return None
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager

    log_path = Path(log_path)
    output = Path(output) if output else log_path.parent/"loss_curve.png"
    history = read_history(log_path)
    # 本地优先中文字体；服务器无中文字体时使用英文，避免图中出现缺字方框。
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if name in available), None)
    chinese = font is not None
    style = {"font.family":font or "DejaVu Sans", "axes.unicode_minus":False,
             "font.size":10, "axes.spines.top":False, "axes.spines.right":False}
    def label(cn, en):
        return cn if chinese else en
    with plt.rc_context(style):
        fig, (ax, lr_ax) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True,
                                      gridspec_kw={"height_ratios":[3, 1.15]})
        for index, epoch in enumerate(sorted({x["epoch"] for x in history.updates})):
            rows = [x for x in history.updates if x["epoch"] == epoch]
            ax.plot([x["step"] for x in rows], [x["loss"] for x in rows], color="#2563eb", lw=1.9,
                    label=label("训练损失：轮内累计平均", "Training: within-epoch cumulative mean") if index==0 else None)
            recent = [x for x in rows if x["window_loss"] is not None]
            if recent:
                ax.plot([x["step"] for x in recent], [x["window_loss"] for x in recent], color="#e89a28", lw=1.1, alpha=.8,
                        label=label("训练损失：最近日志区间平均", "Training: recent logging-window mean") if index==0 else None)
        ends = [x for x in history.epochs if x["step"] is not None]
        if ends:
            ax.scatter([x["step"] for x in ends], [x["train_loss"] for x in ends], color="#2563eb", s=38, zorder=4,
                       label=label("每轮最终训练均值", "Final training mean per epoch"))
            last = ends[-1]
            ax.annotate(f"Train {last['train_loss']:.6f}", (last["step"], last["train_loss"]),
                        xytext=(-10,14), textcoords="offset points", ha="right", color="#1e40af")
        if history.validations:
            rows = history.validations
            ax.plot([x["step"] for x in rows], [x["loss"] for x in rows], marker="D", ms=6, color="#dc4446",
                    linestyle="--" if len(rows)>1 else "None", lw=1.3,
                    label=label("验证损失：实际测量点", "Validation: measured checkpoints"))
            last = rows[-1]
            ax.annotate(f"Val {last['loss']:.6f}", (last["step"],last["loss"]), xytext=(-10,-19),
                        textcoords="offset points", ha="right", color="#b91c1c")
        ax.set_ylabel(label("交叉熵损失", "Cross-entropy loss"))
        ax.set_title(label("损失变化（不对原始曲线额外平滑）", "Loss history (no extra smoothing)"), loc="left", fontsize=12, pad=12)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        lr_rows = [x for x in history.updates if x["lr"] is not None]
        if lr_rows:
            lr_ax.plot([x["step"] for x in lr_rows], [x["lr"] for x in lr_rows], color="#8b5cf6", lw=1.8)
            lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            lr_ax.set_ylim(bottom=0)
        else:
            lr_ax.text(.5,.5,label("该日志未记录学习率", "Learning rate was not logged"), transform=lr_ax.transAxes, ha="center")
        lr_ax.set_ylabel(label("学习率", "Learning rate"))
        lr_ax.set_xlabel(label("优化器更新次数（global_step）", "Optimizer updates (global_step)"))
        for axis in (ax, lr_ax):
            axis.grid(True, alpha=.18, color="#94a3b8")
            axis.set_axisbelow(True)
            axis.margins(x=.045)
        fig.suptitle(title or log_path.parent.name, fontsize=16, fontweight="bold", x=.09, ha="left", y=.98)
        notes = [label("累计均值会自然变平，不能代替局部损失或验证趋势。", "Cumulative means naturally flatten; inspect local loss and validation separately.")]
        if history.updates and history.updates[0]["step"] > 1:
            notes.append(label(f"首个日志点为 step {history.updates[0]['step']}，没有补绘缺失点。", f"First logged step is {history.updates[0]['step']}; missing points are not fabricated."))
        if len(history.validations)==1:
            notes.append(label("仅有 1 个验证点，无法判断验证趋势或过拟合。", "Only one validation point: no validation trend or overfitting conclusion."))
        fig.text(.09,.018,"\n".join(notes),fontsize=8.5,color="#475569",va="bottom")
        fig.tight_layout(rect=(0,.11,1,.94), h_pad=1.7)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, facecolor="white")
        plt.close(fig)
    print("损失图：", output)
    for note in history.notes:
        print("绘图说明：", note)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="同一实验的 train.log 或 metrics.jsonl")
    parser.add_argument("--output", type=Path, help="默认输出到日志旁的 loss_curve.png；也支持 .svg")
    parser.add_argument("--title")
    args = parser.parse_args()
    plot_loss(args.log, args.output, args.title)


if __name__ == "__main__":
    main()
