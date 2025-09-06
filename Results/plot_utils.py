import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import re
from scipy.optimize import curve_fit
from matplotlib.ticker import (
    MaxNLocator, AutoMinorLocator,
    LogLocator, LogFormatter
)

sns.set_theme(
    style="whitegrid",        # white background + grid
    rc={                       # rc overrides
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "grid.color":       "#dddddd",
        "font.size":        14,    # base font size
        "axes.titlesize":   16,    # title
        "axes.labelsize":   14,    # x/y labels
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  12,
        "legend.title_fontsize": 13,
    }
)

params = {
    'meta-llama/Llama-3.1-8B-Instruct': 8,
    'Qwen/Qwen3-8B':8,
    'google/gemma-2-9b-it':9,
    'Qwen/Qwen3-14B':14,
    'mistralai/Ministral-8B-Instruct-2410':8,
    'Qwen/QwQ-32B':32,
    'Qwen/Qwen3-32B':32,
    'mistralai/Mistral-Small-3.1-24B-Instruct-2503': 24,
    'google/gemma-2-27b-it':27,
    'meta-llama/Llama-3.3-70B-Instruct':70,
    'nvidia/Llama-3_3-Nemotron-Super-49B-v1':49,
    'Qwen/Qwen2-72B-Instruct':72,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 56,
    'mistralai/Mixtral-8x22B-Instruct-v0.1': 141,
    'meta-llama/Llama-4-Scout-17B-16E-Instruct': 109,
    'meta-llama/Llama-4-Scout-17B-16E':109,
    'Qwen/Qwen3-235B-A22B-FP8' : 235,
    'deepseek-ai/DeepSeek-R1-0528' : 670,
}

active_params = {
    'mistralai/Mixtral-8x7B-Instruct-v0.1':   12.9,  # Mixtral‑8x7B uses 12.9 B active parameters per token :contentReference[oaicite:0]{index=0}
    'mistralai/Mixtral-8x22B-Instruct-v0.1':  39,    # Mixtral‑8x22B uses 39 B active parameters per token :contentReference[oaicite:1]{index=1}
    'meta-llama/Llama-4-Scout-17B-16E-Instruct': 17,  # Llama 4 Scout activates 17 B parameters per pass :contentReference[oaicite:2]{index=2}
    'Qwen/Qwen3-235B-A22B-FP8':                  22,   # Qwen3‑235B‑A22B activates 22 B parameters per inference :contentReference[oaicite:4]{index=4}
    'deepseek-ai/DeepSeek-R1-0528':              37,   # DeepSeek‑R1 activates 37 B parameters per forward pass :contentReference[oaicite:5]{index=5}
}

def shorten(lab):
    if lab == 'bfloat16':
        return 'bf16'
    if lab == 'float16':
        return 'fp16'
    else:
        return lab

def shorten_hw_name(hw_name: str) -> str:
    """
    Shorten a hardware description string to "<Vendor> <Model>",
    where <Model> is the first token (or subtoken) containing a digit.
    Special case: for AMD MI250X/MI250, return "MI250X" or "MI250" only.
    """
    # Special case for AMD MI250X/MI250
    if hw_name == 'AMD Instinct MI250X/MI250':
        return 'AMD MI250X'

    tokens = hw_name.split()
    vendor = tokens[0]

    # Flatten subtokens (split on "-") so we catch things like "A100-SXM4-80GB"
    subtokens = []
    for tok in tokens[1:]:
        if "-" in tok:
            subtokens.extend(tok.split("-"))
        else:
            subtokens.append(tok)

    # Find the first subtoken with at least one digit
    for st in subtokens:
        if re.search(r"\d", st):
            model = st
            break
    else:
        # fallback to the second token if nothing matched
        model = tokens[1] if len(tokens) > 1 else ""

    return f"{vendor} {model}"

# Shorten model names
def shorten_name(name):
    return (
        name.split('/')[-1]
        .replace('-', ' ')
        .replace('_','.')
        .replace('v1', '')
        .replace(' FP8','')
        .replace('v0.1', '')
        .replace('Instruct', '')
        .replace('Super ', '')
        .replace('it', '')
        .replace('Small ', '')
        .replace('2503', '')
        .replace('2410', '')
        .replace('0528', '')
        .replace(' 17B 16E', '')
        .replace('Nemotron', '')
        .strip()
    )

# Mark Mixture-of-Experts models
MoEs = [
    'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'Qwen/Qwen3-235B-A22B-FP8',
    'deepseek-ai/DeepSeek-R1-0528',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct'
]

def is_moe(name):
    return name in MoEs

def process(df):
    df.loc[:, 'MoE'] = df.loc[:, 'Model Name'].apply(is_moe)

    # Total tokens
    df.loc[:, 'Tokens Total'] = df.loc[:, 'In Tokens Total'] + df.loc[:, 'Out Tokens Total']

    # Energy per token (J/1000)
    df.loc[:, 'Active Energy/Tok (in+out) mJ'] = df.loc[:, 'active_energy'] / df.loc[:, 'Tokens Total'] * 1000
    df.loc[:, 'Active Energy/Tok (in) mJ']     = df.loc[:, 'active_energy'] / df.loc[:, 'In Tokens Total']  * 1000
    df.loc[:, 'Active Energy/Tok (out) mJ']    = df.loc[:, 'active_energy'] / df.loc[:, 'Out Tokens Total'] * 1000

    #Tokens per energy
    df.loc[:, 'Tok per kJ (in+out)'] = df.loc[:, 'Tokens Total'] / df.loc[:, 'active_energy'] * 1000 
    df.loc[:, 'Tok per kJ (in)']     = df.loc[:, 'In Tokens Total'] / df.loc[:, 'active_energy'] * 1000
    df.loc[:, 'Tok per kJ (out)']    = df.loc[:, 'Out Tokens Total'] / df.loc[:, 'active_energy'] * 1000

    # Derived sizes
    df.loc[:, 'Total Batch Size'] = df.loc[:, 'Batch Size'] * df.loc[:, 'DP Size']
    df.loc[:, 'Total GPUs']       = (
        df.loc[:, 'TP Size'] * df.loc[:, 'PP Size'] * df.loc[:, 'EP Size'] * df.loc[:, 'DP Size']
    )

    df.loc[:, 'Batch Size per GPU'] = df.loc[:, 'Total Batch Size'] / df.loc[:, 'Total GPUs']

    # Number of parameters
    df.loc[:, '# params'] = df.apply(lambda x: params[x['Model Name']], axis=1)
    df.loc[:, 'active params'] = df.apply(lambda x: active_params[x['Model Name']] if x['MoE'] else params[x['Model Name']], axis=1)

    # Throughput per watt
    df.loc[:, 'Throughput per Watt (in+out)'] = df.loc[:, 'Throughput (in+out) tok/s'] / df.loc[:, 'active_power_avg']
    df.loc[:, 'Throughput per Watt (in)'] = df.loc[:, 'Throughput (in) tok/s'] / df.loc[:, 'active_power_avg']
    df.loc[:, 'Throughput per Watt (out)'] = df.loc[:, 'Throughput (out) tok/s'] / df.loc[:, 'active_power_avg']

    df.loc[:, 'Throughput per User'] = df.loc[:, 'Throughput (in+out) tok/s'] / df.loc[:, 'Batch Size']

    df.loc[:, 'Model Name Short'] = df.loc[:, 'Model Name'].apply(shorten_name)

    df.loc[:, 'Hardware Type Short'] = df.loc[:, 'Hardware type'].apply(shorten_hw_name)

    # Config columns
    df.loc[:, 'Config'] = df.apply(
        lambda r: f"TP: {r['TP Size']}, DP: {r['DP Size']}", axis=1
    )

    df.loc[:, 'Config EP'] = df.apply(
        lambda r: f"TP: {r['TP Size']}, EP: {r['EP Size']}", axis=1
    )

    df.loc[:, 'Config HW'] = df.apply(
        lambda r: f"{r['Hardware Type Short']}\nTP: {r['TP Size']}, DP: {r['DP Size']}", axis=1
    )

    df.loc[:, 'HW count'] = df.apply(
        lambda r: f"{r['Total GPUs']}x{r['Hardware Type Short']}", axis=1
    )

    df.loc[:, 'Config Precision'] = df.apply(
        lambda r: f"TP: {r['TP Size']}, DP: {r['DP Size']} \n{r['Precision']}", axis=1
    )

    df.loc[:, 'Model Precision'] = df.apply(
        lambda r: f"{r['Model Name Short']} ({r['Precision']})", axis=1
    )

    df.loc[:, 'TP Precision'] = df.apply(
        lambda r: f"{r['TP Size']} ({r['Precision']})", axis=1
    )

def remove_duplicates_and_sort(df):
    df.sort_values(by=['# params','Hardware type','TP Size', 'DP Size', 'EP Size', 'Precision'], ascending=True, inplace=True)

    print("Before pruning: ", len(df))

    df = df.drop_duplicates(
        subset=['Hardware type', 'Model Name', 'TP Size', 'PP Size', 'DP Size', 'EP Size', 'Batch Size', 'Precision'],
        keep='last'
    )

    print("After pruning: ", len(df))

    df = df[(df['Total Batch Size'].isna()) | (df['Total Batch Size'] <= 1024)]

    return df

def annotate_nonzero(axis, digits=0):
    for p in axis.patches:
        h = p.get_height()
        if h <= 0:
            continue
        axis.annotate(
            f"{h:.{digits}f}",
            xy=(p.get_x() + p.get_width() / 2, h),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=20
        )

def pareto_barrier(df, model_name, x, y, hw=None, title=None, palette= 'viridis', save = False):
    model_df = df[(df['Model Name'] == model_name)].copy()

    short_name = shorten_name(model_name)
    
    plt.figure(figsize=(12, 6))
    
    if hw:
        model_df = model_df[model_df['Hardware type'] == hw]
        sns.lineplot(
            data=model_df,
            x=x,
            y=y,
            hue='Config',
            markers=True,
            dashes=True,
            ci=None,
            palette=palette,         
        )

    else:
        sns.lineplot(
            data=model_df,
            x=x,
            y=y,
            hue='Hardware Type Short',
            markers=True,
            dashes=True,
            style='Precision',
            ci=None,
            palette=palette,
        )

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{short_name} {x} / {y} tradeoff')
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(title='Configurations', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save:
        plt.savefig(f'./Plots/{short_name} {y.split(' ')[0]} BS.pdf')
    plt.show()

def plot_metric(df, model_name, metric, hw=None, title=None, palette= 'viridis', save = False):
    model_df = df[(df['Model Name'] == model_name)].copy()

    short_name = shorten_name(model_name)
    
    plt.figure(figsize=(12, 6))
    
    if hw:
        model_df = model_df[model_df['Hardware type'] == hw]
        sns.lineplot(
            data=model_df,
            x='Total Batch Size',
            y=metric,
            hue='Config',
            markers=True,
            dashes=True,
            ci=None,
            palette=palette,         
        )

    else:
        sns.lineplot(
            data=model_df,
            x='Total Batch Size',
            y=metric,
            hue='Hardware Type Short',
            markers=True,
            dashes=True,
            style='Precision',
            ci=None,
            palette=palette,
        )

    plt.xlabel('Batch Size')
    plt.ylabel(metric)
    plt.title(f'{short_name} {metric}')
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(title='Configurations', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save:
        plt.savefig(f'./Plots/{short_name} {metric.split(' ')[0]} BS.pdf')
    plt.show()

def plot_metric_comparison(df, model_names, metric, hw=None,
                           title=None, palette='viridis',
                           save=False, save_path='./Plots'):
    """
    1×4 grid of log‐scaled line plots, with one compact legend
    just outside the right of the figure, using consistent entries/order.
    """
    if len(model_names) != 4:
        raise ValueError("Please pass exactly four model names.")

    df = df.copy()
    if hw is None:
        df['Precision'] = df.apply(
            lambda r: r['Precision']
                      + (f" ({r['TP Size']} Tiles)" if int(r['TP Size']) > 1 else ""),
            axis=1
        )
        df["Hardware"] = df["Hardware Type Short"]

    if hw:
        hue, style = 'Config', None
        hue_order = sorted(df[hue].unique())
        style_order = None
    else:
        hue, style = 'Hardware', 'Precision'
        hue_order = sorted(df[hue].unique())
        style_order = sorted(df[style].unique())

    # 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(36, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (ax, model_name) in enumerate(zip(axes, model_names)):
        sub = df[df['Model Name'] == model_name]
        if hw:
            sub = sub[sub['Hardware type'] == hw]

        short = shorten_name(model_name)
        sns.lineplot(
            data=sub,
            x='Total Batch Size', y=metric,
            hue=hue, style=style,
            hue_order=hue_order,
            style_order=style_order,
            palette=palette,
            markers=True, dashes=True, ci=None,
            ax=ax,
            legend=(i == 0)
        )

        # LOG SCALE + ENHANCED GRID
        ax.set_yscale('log')
        ax.set_axisbelow(True)
        ax.grid(which='major', linestyle='--', linewidth=0.8, alpha=0.8)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

        # X‐axis ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        # Y‐axis ticks: more majors + dense minors with labels
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0,
                       subs=np.arange(2, 10) * 0.1,
                       numticks=50)
        )
        ax.yaxis.set_minor_formatter(LogFormatter(base=10.0, labelOnlyBase=False))

        ax.tick_params(axis='y', which='major', length=6)
        ax.tick_params(axis='y', which='minor', length=4, labelsize='small')

        ax.set_title(short)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel(metric)

    # extract legend handles/labels
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()

    # tighten right margin & pull legend close to last plot
    fig.subplots_adjust(right=0.85, bottom=0.15)
    fig.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(0.86, 0.90),
        ncol=1,
        fontsize='small',
        handlelength=1.2,
        borderaxespad=0.0,
        frameon=True,
    )

    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    if save:
        fname = f"{(title or metric).replace('/', '_').replace(' ', '_')}_comparison.pdf"
        fig.savefig(f"{save_path}/{fname}", bbox_inches='tight')

    plt.show()

def compare_metric_bar(df, model_name, metric, batch_size=128, title=None, palette='viridis', save=False):
    # filter down to just this model, batch size, and the two precisions
    sub = df[
        (df['Model Name'] == model_name) &
        (df['Total Batch Size'] == batch_size) &
        (df['Precision'].isin(['bfloat16', 'fp8']))
    ].copy()
    
    if sub.empty:
        print(f"No data available for {model_name} at batch size {batch_size} with specified precisions.")
        return
    
    # ensure consistent ordering of precisions
    sub['Precision'] = pd.Categorical(sub['Precision'],
                                      categories=['bfloat16', 'fp8'],
                                      ordered=True)
    
    if title is None:
        title = metric

    # draw grouped barplot
    plt.figure(figsize=(4*len(df['Hardware Type Short'].unique())+2, 6))
    ax = sns.barplot(
        data=sub,
        x='Hardware Type Short',
        y=metric,
        hue='Precision',
        hue_order=['bfloat16', 'fp8'],
        ci=None,
        palette=palette,
        edgecolor='black',
        linewidth=1,
    )
    
    # annotate each bar with its height
    annotate_nonzero(ax)
    short_name = shorten_name(model_name)
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.set_title(f'{short_name} {title}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    if save:
        fname = f"./Plots/{short_name}_{title}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    plt.show()

def compare_metric_bar2(df, model_name, metric, metric2=None,
                        batch_size=128, palette='viridis', digits=0,
                        title=None, save=False):
    # 1) Filter
    sub = df[
        (df['Model Name'] == model_name) &
        (df['Total Batch Size'] == batch_size) &
        (df['Precision'].isin(['bfloat16', 'fp8']))
    ].copy()
    
    if sub.empty:
        print(f"No data for {model_name} @ batch size {batch_size}.")
        return
    if title is None:
        title = metric

    # 2) Order precisions
    sub['Precision'] = pd.Categorical(
        sub['Precision'],
        categories=['bfloat16', 'fp8'],
        ordered=True
    )

    # 3) Plot
    plt.figure(figsize=(24,8))
    ax = sns.barplot(
        data=sub, x='Hardware Type Short', y=metric,
        hue='TP Precision', palette=palette, ci=None,
        alpha=0.7, edgecolor='black', linewidth=1
    )

    if metric2:
        ax2 = sns.barplot(
            data=sub, x='Hardware Type Short', y=metric2,
            hue='TP Precision', palette=palette, ci=None,
            alpha=1, edgecolor='black', linewidth=1,
            ax=ax
        )

    annotate_nonzero(ax, digits=digits)
    if metric2:
        annotate_nonzero(ax2, digits=digits)


    # 6) Final touches
    short_name = shorten_name(model_name)
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.set_title(f"{short_name} {title}")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    plt.tight_layout()

    if save:
        fname = f"./Plots/{short_name.replace('/',' ')}_{title.replace('/',' ')}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved figure to {fname}")

    plt.show()

def compare_metric_bar3(df, model_name, metric, metric2=None,precision='fp8',
                        batch_size=128, palette='viridis',
                        title=None, save=False):
    # 1) Filter
    sub = df[
        (df['Model Name'] == model_name) &
        (df['Total Batch Size'] == batch_size) &
        (df['Precision'] == precision)
    ].copy()

    sub.loc[:, 'Parallel Mode'] = df.apply(
        lambda r: 'Expert Parallel' if r['EP Size'] > 1 else 'Tensor Parallel', axis=1
    )
    
    if sub.empty:
        print(f"No data for {model_name} @ batch size {batch_size}.")
        return
    if title is None:
        title = metric

    # 3) Plot
    plt.figure(figsize=(4*len(df['HW count'].unique())+2, 6))
    ax = sns.barplot(
        data=sub, x='HW count', y=metric,
        hue='Parallel Mode', palette=palette, ci=None,
        alpha=0.7, edgecolor='black', linewidth=1
    )

    if metric2:
        ax2 = sns.barplot(
            data=sub, x='HW count', y=metric2,
            hue='Parallel Mode', palette=palette, ci=None,
            alpha=1, edgecolor='black', linewidth=1,
            ax=ax
        )

        ax2.legend_.remove()
    
    annotate_nonzero(ax)
    if metric2:
        annotate_nonzero(ax2)

    # Create a manual legend using the first plot handles
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # Avoid duplicates
    ax.legend(
        unique.values(), unique.keys(),
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0
    )

    # 6) Final touches
    short_name = shorten_name(model_name)
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.set_title(f"{short_name} ({precision}) {title}")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save:
        fname = f"./Plots/{short_name}_{title}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved figure to {fname}")

    plt.show()

def compare_metric_bar_with_baseline(df, model_name, metric, batch_size=128, palette='viridis'):
    sub = df[
        (df['Model Name'] == model_name) &
        (df['Total Batch Size'] == batch_size) &
        (df['Precision'].isin(['bfloat16', 'fp8']))
    ].copy()

    # 2) Further filter to your chosen “baseline” row
    df_baseline = sub[
        (sub['Hardware type'] == 'NVIDIA A100-SXM4-80GB') &
        (sub['Precision']   == 'bfloat16')
    ]
    try:
        # 3) Extract the baseline *metric* (not the whole row!)
        baseline = df_baseline[metric].iloc[0]

    except:
        return
    
    if sub.empty:
        print(f"No data available for {model_name} at batch size {batch_size} with specified precisions.")
        return
    
    # ensure consistent ordering of precisions
    sub['Precision'] = pd.Categorical(sub['Precision'],
                                      categories=['bfloat16', 'fp8'],
                                      ordered=True)
    
    # draw grouped barplot
    plt.figure(figsize=(4*len(df['Hardware Type Short'].unique())+2, 6))
    ax = sns.barplot(
        data=sub,
        x='Hardware Type Short',
        y=metric,
        hue='Precision',
        hue_order=['bfloat16', 'fp8'],
        ci=None,
        palette=palette,
        alpha=1, edgecolor='black', linewidth=1,
    )
    
    # annotate each bar with its height
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}",
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=12)
        
        ax.annotate(f"(+{(height / baseline - 1)*100:.0f}%)" if height >= baseline else f"({(height / baseline - 1)*100:.0f}%)",
            (p.get_x() + p.get_width() / 2, height-200),
            ha='center', va='top', fontsize=12, fontweight='bold')
        
    short_name = shorten_name(model_name)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(f'{short_name} {metric}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def tp_vs_ep_metric2(df, model_name, metric, hw=None, bs=None, palette='viridis'):

    # filter the dataframe
    if hw:
        model_df = df[(df['Model Name'] == model_name) & (df['Hardware type'] == hw)].copy()
    else:
        model_df = df[df['Model Name'] == model_name].copy()


    # build a per-row “config” string (fixed quoting and using ds_df, not df)
    model_df["config"] = model_df.apply(
        lambda r: f"{r['Hardware Type Short']} EP: {r['EP Size']}" if r['EP Size'] > 1 else f"{r['Hardware Type Short']} TP: {r['TP Size']}", axis=1
    )
    if bs:
        unbounded_df = model_df[model_df['Batch Size'] == bs].copy()
    else:
        unbounded_df = model_df[model_df['Batch Size'].isna()].copy()

    # set up the figure size for readability
    plt.figure(figsize=(4*len(df['Hardware Type Short'].unique())+2, 6))

    # draw the barplot
    sns.barplot(
        data=unbounded_df,
        x='config',
        y=metric,
        ci=None,
        palette=palette
    )

    # rotate x-labels so they don’t overlap
    plt.xticks(rotation=45, ha='right')

    # add axis labels and title
    #plt.xlabel("Configuration")
    plt.ylabel(metric)
    short_name = shorten_name(model_name)
    plt.title(f'{short_name} {metric}')
    plt.tight_layout()
    plt.show()

def tp_vs_ep_metric(df, model_name, metric, hw=None, palette='viridis', save=False):

    if hw:
        model_df = df[(df['Model Name'] == model_name) & (df['Hardware type'] == hw)].copy()
    else:
        model_df = df[(df['Model Name'] == model_name)].copy()

    if len(model_df) == 0:
        print("Empty")
        return

    model_df['Batch Size'] = model_df['Batch Size'].fillna(256)

    model_df.loc[:, 'Parallel Mode'] = df.apply(
        lambda r: "Expert Parallel" if int(r['EP Size']) > 1 else "Tensor Parallel", axis=1
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=model_df,
        x='Batch Size',
        y=metric,
        hue='HW count',
        style='Parallel Mode',
        markers=True,
        dashes=True,
        ci=None,
        palette=palette,
    )

    plt.xlabel('Batch Size')
    plt.ylabel(metric)
    short_name = shorten_name(model_name)
    plt.title(f'{short_name} {metric}')
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(title='# GPUs  (style=EP)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save:
        fname = f"./Plots/{short_name}_ep_tp.pdf"
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    plt.show()

def compare_metric_multi_gpus(df, model_name, metric, hw, batch_size=128, title=None, palette='viridis', save=False):
    # filter down to just this model, batch size, and the two precisions
    sub = df[
        (df['Model Name'] == model_name) &
        (df['Total Batch Size'] == batch_size) &
        (df['Precision'].isin(['bfloat16', 'fp8'])) &
        (df['Hardware type'] == hw)
    ].copy()
    
    if sub.empty:
        print(f"No data available for {model_name} at batch size {batch_size} with specified precisions.")
        return
    
    # ensure consistent ordering of precisions
    sub['Precision'] = pd.Categorical(sub['Precision'],
                                      categories=['bfloat16', 'fp8'],
                                      ordered=True)
    
    if title is None:
        title = metric

    # draw grouped barplot
    plt.figure(figsize=(4*len(df['Hardware Type Short'].unique())+2, 6))
    ax = sns.barplot(
        data=sub,
        x='Config',
        y=metric,
        hue='Precision',
        hue_order=['bfloat16', 'fp8'],
        ci=None,
        palette=palette,
        edgecolor='black',
        linewidth=1,
    )
    
    # annotate each bar with its height
    annotate_nonzero(ax)
    short_name = shorten_name(model_name)
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.set_title(f'{sub['Hardware Type Short'].iloc[0]} scaling for {sub['Model Name Short'].iloc[0]}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    if save:
        fname = f"./Plots/{short_name}_{title}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    plt.show()

def compare_metric_multi_gpus_eff(df, df_base, model_name, metric, hw, batch_size=128, title=None, palette='viridis', save=False):
    # filter down to just this model, batch size, and the two precisions
    sub = df[
        (df['Model Name'] == model_name) &
        (df['Total Batch Size'] == batch_size) &
        (df['Precision'].isin(['bfloat16', 'fp8'])) &
        (df['Hardware type'] == hw)
    ].copy()
    
    gpus = sub['Total GPUs'].iloc[0]
    try:
        base_bf16 = df_base[
            (df_base['Model Name'] == model_name) &
            (df_base['Total Batch Size'] == batch_size / gpus) &
            (df_base['Precision'] == 'bfloat16') &
            (df_base['Hardware type'] == hw)
        ][metric].iloc[0]

        base_fp8 = df_base[
            (df_base['Model Name'] == model_name) &
            (df_base['Total Batch Size'] == batch_size / gpus) &
            (df_base['Precision'] == 'fp8') &
            (df_base['Hardware type'] == hw)
        ][metric].iloc[0]
    except:
        return

    eff_col = metric + ' efficiency'

    sub[eff_col] = np.nan

    # assign *in place* via .loc
    sub.loc[sub['Precision']=='fp8',    eff_col] = sub.loc[sub['Precision']=='fp8',    metric] / (base_fp8 * gpus) *  100
    sub.loc[sub['Precision']=='bfloat16', eff_col] = sub.loc[sub['Precision']=='bfloat16', metric] / (gpus * base_bf16) * 100
    
    if title is None:
        title = metric

    # draw grouped barplot
    plt.figure(figsize=(4*len(df['Hardware Type Short'].unique())+2, 6))
    ax = sns.barplot(
        data=sub,
        x='Config',
        y=metric,
        hue='Precision',
        hue_order=['bfloat16', 'fp8'],
        ci=None,
        palette=palette,
        edgecolor='black',
        linewidth=1,
    )
    
    # annotate each bar with its height
    annotate_nonzero(ax)
    short_name = shorten_name(model_name)
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.set_title(f'{sub['Hardware Type Short'].iloc[0]} scaling for {sub['Model Name Short'].iloc[0]}')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    if save:
        fname = f"./Plots/{short_name}_{title}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    plt.show()

def compare_metric_multi_gpus_eff2(df, model_name, metric, metric2 = None, hw = None, batch_size=128, title=None, palette='viridis', save=False):
    # filter down to just this model, batch size, and the two precisions
    sub = df[
        (df['Model Name'] == model_name) &
        (df['Batch Size per GPU'] == batch_size) &
        (df['Precision'].isin(['bfloat16', 'fp8']))
    ].copy()

    if hw:
        sub = sub[(df['Hardware type'] == hw)]

    if len(sub) == 0:
        return
    
    if title is None:
        title = metric

    if hw:
        # draw grouped barplot
        plt.figure(figsize=(4*len(df['Config'].unique())+2, 6))
        ax = sns.barplot(
            data=sub,
            x='Config',
            y=metric,
            hue='Precision',
            hue_order=['bfloat16', 'fp8'],
            ci=None,
            palette=palette,
            edgecolor='black',
            linewidth=1,
        )
        ax.set_title(f'{sub['Hardware Type Short'].iloc[0]} scaling for {sub['Model Name Short'].iloc[0]}')

    else:
        # draw grouped barplot
        plt.figure(figsize=(4*len(df['Config HW'].unique())+2, 6))
        ax = sns.barplot(
            data=sub,
            x='Config HW',
            y=metric,
            hue='Precision',
            hue_order=['bfloat16', 'fp8'],
            ci=None,
            palette=palette,
            edgecolor='black',
            alpha=0.7,
            linewidth=1,
        )

        if metric2:
            ax2 = sns.barplot(
                data=sub,
                x='Config HW',
                y=metric2,
                hue='Precision',
                hue_order=['bfloat16', 'fp8'],
                ci=None,
                palette=palette,
                edgecolor='black',
                linewidth=1,
            )
            annotate_nonzero(ax2)

        ax.set_title(f'{title} {sub['Model Name Short'].iloc[0]}')
    
    # annotate each bar with its height
    annotate_nonzero(ax)

    short_name = shorten_name(model_name)
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()

    if save:
        if hw:
            fname = f"./Plots/{hw}_{short_name}_{title.split('/')[0]}.pdf"
        else:
            fname = f"./Plots/{short_name}_{title.split('/')[0]}.pdf"
        plt.savefig(fname, bbox_inches='tight')
        print(f"Saved figure to {fname}")
    plt.show()

def plot_efficiency_heatmap(df, df_base, model_name, metric, batch_size=128, precisions=['bfloat16', 'fp8'], cmap='viridis'):
    """
    Draws side-by-side heatmaps of efficiency for each precision,
    with Hardware Type Short on the y-axis and TP size on the x-axis.
    """
    # Filter to only the specified model, batch size, and precisions
    sub = df[
        (df['Model Name'] == model_name) &
        (df['Total Batch Size'] == batch_size) &
        (df['Precision'].isin(precisions))
    ].copy()

    # Compute efficiency for each row
    eff_col = metric + ' efficiency'
    sub[eff_col] = np.nan

    try:
        for hw in sub['Hardware type'].unique():
            for prec in precisions:
                mask = (sub['Hardware type'] == hw) & (sub['Precision'] == prec)
                if not sub[mask].empty:
                    gpus = sub.loc[mask, 'Total GPUs'].iloc[0]
                    # Find corresponding base metric
                    base = df_base[
                        (df_base['Model Name'] == model_name) &
                        (df_base['Precision'] == prec) &
                        (df_base['Hardware type'] == hw) &
                        (df_base['Total Batch Size'] == batch_size / gpus)
                    ][metric].iloc[0]
                    sub.loc[mask, eff_col] = sub.loc[mask, metric] / (base * gpus) * 100
    
    except:
        return

    # Prepare plotting
    precisions_found = sub['Precision'].unique()
    n = len(precisions_found)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, prec in zip(axes, precisions_found):
        data = sub[sub['Precision'] == prec].pivot(
            index='Hardware Type Short',
            columns='TP Size',
            values=eff_col
        )
        sns.heatmap(data, annot=True, fmt='.1f', cmap=cmap, ax=ax)
        ax.set_title(f"{prec} Efficiency")
        ax.set_xlabel("TP size")
        ax.set_ylabel("Hardware Type Short")

    plt.tight_layout()
    plt.show()


def plot_trend(df, metric, hw=None, batch_size=128, title=None, palette= 'viridis', save = False):    
    plt.figure(figsize=(12, 6))
    
    df_bs = df[df['Batch Size'] == 128]

    if hw:
        df = df[df['Hardware type'] == hw]
        sns.lineplot(
            data=df_bs,
            x='# params',
            y=metric,
            hue='Config',
            markers=True,
            dashes=True,
            ci=None,
            palette=palette         
        )

    else:
        sns.lineplot(
            data=df_bs,
            x='# params',
            y=metric,
            hue='Hardware Type Short',
            markers=True,
            dashes=True,
            style='Precision',
            ci=None,
            palette=palette         
        )

    plt.xscale('log')
    plt.xlabel('Parameters (Billions)')
    plt.ylabel(metric)
    plt.title(f'{metric} across model sizes')
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.legend(title='Configurations', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save:
        plt.savefig(f'./Plots/Sizes {metric.split(' ')[0]}.pdf')
    plt.show()