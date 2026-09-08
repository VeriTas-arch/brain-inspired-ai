"""Audit and report the fixed three-task experiment without combining game scales."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.study_gpm import ROOT, file_hash, normalized_auc, write_json
from scripts.study_three_tasks import METHODS, TASKS


def main():
    output = ROOT / "outputs/three_tasks_v1/seed-61001"
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["protocol"]
    assert json.loads((output / "completion.json").read_text())["status"] == "completed"
    for name, expected in manifest["source_sha256"].items():
        assert file_hash(output / "source_snapshot" / name) == expected, name
        assert file_hash(ROOT / name) == expected, name
    assert (
        file_hash(ROOT / manifest["settings"]["anchor_dir"] / "anchor.pt")
        == manifest["anchor_sha256"]
    )
    summaries, histories = {}, {}
    for method in METHODS:
        directory = output / method
        summary = json.loads((directory / "summary.json").read_text())
        assert len(summary["score_matrix"]) == 3
        for row in summary["score_matrix"]:
            assert set(row) == set(TASKS)
            assert all(len(v) == config["eval_episodes"] for v in row.values())
        for stage in (2, 3):
            history = json.loads((directory / f"stage-{stage}/history.json").read_text())["history"]
            assert [row["steps"] for row in history] == list(
                range(0, config["new_steps"] + 1, config["eval_interval"])
            )
            assert history[0]["scores"] == summary["score_matrix"][stage - 2]
            assert history[-1]["scores"] == summary["score_matrix"][stage - 1]
            assert all(
                len(scores) == config["eval_episodes"]
                for row in history
                for scores in row["scores"].values()
            )
            assert np.isclose(
                normalized_auc(history, TASKS[stage - 1]), summary["stages"][stage - 2]["new_auc"]
            )
            assert summary["stages"][stage - 2]["old_heads_unchanged"]
            if method == "gpm":
                for row in history:
                    assert (
                        row["projection"]["optimizer_steps"]
                        == row["steps"] * config["update_epochs"] // config["minibatch_size"]
                    )
            histories[method, stage] = history
            if method == "ewc":
                checkpoint = torch.load(
                    directory / f"stage-{stage}/final.pt", map_location="cpu", weights_only=True
                )
                assert checkpoint["current_task_id"] == stage
                assert len(checkpoint["task_fisher"]) == stage
                del checkpoint
        summaries[method] = summary
    assert all(
        summaries[m]["score_matrix"][0] == summaries["finetune"]["score_matrix"][0] for m in METHODS
    )
    shared_heads = None
    for method in METHODS:
        checkpoint = torch.load(
            output / method / "stage-2/final.pt", map_location="cpu", weights_only=True
        )
        heads = {kind: checkpoint["agent"][kind][TASKS[2]] for kind in ("actors", "critics")}
        if shared_heads is not None:
            for kind in heads:
                for name, value in heads[kind].items():
                    torch.testing.assert_close(value, shared_heads[kind][name], rtol=0, atol=0)
        shared_heads = heads
        del checkpoint
    # Independent checks of nested subspaces and their protection during both stages.
    directory = output / "gpm"
    previous = torch.load(directory / "subspaces_after_pong.pt", weights_only=True)
    ranks = [{name: entry["rank"] for name, entry in previous.items()}]
    for stage in (2, 3):
        current = torch.load(directory / f"stage-{stage}/subspaces.pt", weights_only=True)
        for name, entry in current.items():
            basis, old = entry["basis"], previous[name]["basis"]
            assert previous[name]["rank"] <= entry["rank"] <= entry["dimension"]
            torch.testing.assert_close(basis.T @ basis, torch.eye(entry["rank"]), atol=2e-5, rtol=0)
            torch.testing.assert_close(basis @ (basis.T @ old), old, atol=2e-5, rtol=0)
            assert entry["captured_energy_fraction"] >= manifest["settings"]["threshold"] - 1e-6
        for entry in summaries["gpm"]["stages"][stage - 2]["projection"]["layers"].values():
            assert entry["projection_relative_error"] < 1e-3
        ranks.append({name: entry["rank"] for name, entry in current.items()})
        previous = current
    events = []
    for line in (ROOT / "logs/three_tasks_v1/three_tasks_seed61001.log").read_text().splitlines():
        if line.startswith('{"task"') and "first_rollout_sha256" in line:
            events.append(json.loads(line))
    hashes = [e["first_rollout_sha256"] for e in events if e["task"] == TASKS[1]]
    assert len(hashes) == 3 and len(set(hashes)) == 1
    assert hashes[0] == "fa60939715c672cae9ab57db5cc189d719e093cef421b281d5f5b52048dfe38f"
    write_json(
        output / "audit.json",
        dict(
            passed=True,
            source_and_anchor_hashes=True,
            exact_budgets_and_episode_counts=True,
            full_ewc_state=True,
            nested_gpm_subspaces=True,
            matched_breakout_first_rollout=hashes[0],
            ranks=ranks,
        ),
    )
    lines = [
        "# 三任务实验结果（seed 61001）",
        "",
        "复用已学好的 Pong 起点，重新训练 Breakout 与 SpaceInvaders，各 524,288 transitions。每项分数为固定 10 个完整回合的确定性原始奖励均值。只作此单 seed、顺序和预算下的描述性比较。",
        "",
        "本轮 GPM 的三个最终分数均最高：Pong 9.2、Breakout 9.0、SpaceInvaders 307。Pong 相对最初学成时下降 1.8 分，Breakout 相对学成时下降 1.1 分；增加第三任务后仍保留了前两个任务的大部分表现。",
        "",
        "但 GPM 的 Breakout AUC 为 5.3125，SpaceInvaders AUC 为 230.375，均低于本轮普通 PPO 和 EWC。结论是本协议下终点兼顾更好，不能说全面改善了学习速度或消除了稳定性与可塑性的取舍。",
        "",
        "EWC 的 Pong 表现在阶段间和第三任务内部出现过明显恢复与回落；遗忘不是单调的。应同时阅读阶段矩阵、原始 history.json 和新任务曲线，不能把单个检查点解释为知识永久丢失。",
        "",
        "| 方法 | 完成阶段 | Pong | Breakout | SpaceInvaders |",
        "|---|---|---:|---:|---:|",
    ]
    for method, summary in summaries.items():
        for stage, row in enumerate(summary["score_matrix"]):
            lines.append(
                f"| {method} | {TASKS[stage]} | "
                + " | ".join(f"{np.mean(row[t]):.2f}" for t in TASKS)
                + " |"
            )
    lines += [
        "",
        "未学任务的分数仅用于观察前向变化。不同游戏原始分数不合并平均。本轮统一比较重新运行的三任务结果，不混用旧两任务终点；现有 seed_everything 固定随机种子但不强制确定性 GPU 内核，首批 rollout 一致不等于整个训练轨迹逐位相同。",
        "",
        "| 方法 | Breakout AUC / budget | SpaceInvaders AUC / budget | Pong 最终减学习后 | Breakout 最终减学习后 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, summary in summaries.items():
        matrix = summary["score_matrix"]
        lines.append(
            f"| {method} | {summary['stages'][0]['new_auc']:.4f} | {summary['stages'][1]['new_auc']:.4f} | {np.mean(matrix[2][TASKS[0]]) - np.mean(matrix[0][TASKS[0]]):+.2f} | {np.mean(matrix[2][TASKS[1]]) - np.mean(matrix[1][TASKS[1]]):+.2f} |"
        )
    lines += [
        "",
        "## GPM 累计保护维数",
        "",
        "| 层 | 输入维数 | Pong 后 | Breakout 后 | SpaceInvaders 后 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, entry in previous.items():
        lines.append(
            f"| {name} | {entry['dimension']} | " + " | ".join(str(r[name]) for r in ranks) + " |"
        )
    lines += [
        "",
        "第三任务训练使用的是 Breakout 后的累计基；SpaceInvaders 后的基在最后评估结束后才构建，供后续任务续接，不参与本轮 SpaceInvaders 训练。各层尚有剩余维数，但维数本身不能证明任意后续任务都可兼容。",
    ]
    lines += [
        "",
        "## 运行开销",
        "",
        "| 方法 | 任务 | 训练秒数（采集、更新与编译） | 训练与评估墙钟秒数 | GPM 边界采集与构基秒数 |",
        "|---|---|---:|---:|---:|",
    ]
    for method, summary in summaries.items():
        for stage in summary["stages"]:
            lines.append(
                f"| {method} | {stage['task']} | {stage['training_seconds']:.1f} | {stage['wall_seconds']:.1f} | {stage.get('boundary_collection_and_basis_seconds', 0):.1f} |"
            )
    lines += [
        "",
        "GPM 每个新任务边界额外采集 32,768 transitions，只用于构基，不计入 PPO 更新预算；墙钟列不含此边界开销，也不含 checkpoint 写入和 EWC Fisher 估计。",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, task in enumerate(TASKS):
        for method in METHODS:
            axes[0, col].plot(
                [1, 2, 3],
                [np.mean(r[task]) for r in summaries[method]["score_matrix"]],
                marker="o",
                label=method,
            )
        axes[0, col].set(
            title=task, xlabel="Completed stage", ylabel="Raw reward", xticks=[1, 2, 3]
        )
    for col, stage in enumerate((2, 3)):
        task = TASKS[stage - 1]
        for method in METHODS:
            h = histories[method, stage]
            axes[1, col].plot(
                [r["steps"] for r in h],
                [np.mean(r["scores"][task]) for r in h],
                marker="o",
                label=method,
            )
        axes[1, col].set(title=f"Learning {task}", xlabel="Task transitions", ylabel="Raw reward")
    for name, entry in previous.items():
        axes[1, 2].plot(
            [1, 2, 3],
            [r[name] / entry["dimension"] for r in ranks],
            marker="o",
            label=f"Layer {name}",
        )
    axes[1, 2].set(
        title="Protected input dimensions",
        xlabel="Completed stage",
        ylabel="Fraction",
        xticks=[1, 2, 3],
        ylim=(0, 1),
    )
    axes[0, 0].legend()
    axes[1, 2].legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output.parent / f"comparison.{suffix}", dpi=160)
    plt.close(fig)
    lines += [
        "",
        f"![comparison]({output.parent / 'comparison.png'})",
        "",
        "审计通过：源码与 anchor 哈希、精确训练检查点、回合数、AUC、旧 head 不变、EWC 多边界状态、GPM 基的嵌套与正交性、实际位移投影误差、Breakout 首批 rollout 一致。见输出目录 audit.json。",
        "",
        "[冻结协议](three_tasks_v1.md)。结果不能证明一般容量冲突，也不能替代其他任务顺序或 seed 的验证。",
    ]
    (ROOT / "experiments/three_tasks_v1_results.md").write_text("\n".join(lines) + "\n")
    (output / "report_source.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "report_sha256.json", {"report_source.py": file_hash(Path(__file__))})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
