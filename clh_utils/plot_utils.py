import torch


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend, num_keys):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, num_keys)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, window1, update_type, values):
    viz.line(
        X=torch.ones((1, len(values))).cpu() * iteration,
        Y=torch.Tensor(values).unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )

