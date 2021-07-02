# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data visualisations for WP1 report

# %% [markdown]
# ### Import libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# %% [markdown]
# ### Parameters for visualisations

# %%
# colour palette for charts

pal = sns.cubehelix_palette(as_cmap=False)
pal2 = sns.cubehelix_palette(as_cmap=True)


# %% [markdown]
# ### Functions

# %%
# strip white space from dataframe


def strip_white_space(df):
    """strip white spaces from records
    args:
        df: dataframe

    Returns:
        df with white spaces stripped
    """
    cols = df.select_dtypes(object).columns
    df[cols] = df[cols].apply(lambda x: x.str.strip())
    return df


# plot pie chart


def pie(x, ax=None, labels=None, title=None, **plt_kwargs):
    """strip white spaces from records
    args:
        x: data
        ax: axis
        labels: pie chart labels
        title: chart title
        **plt_kwargs: keyword args

    Returns:
        pie charts
    """
    ax = ax
    plt.pie(x=x, autopct="%.1f%%", labels=labels, **plt_kwargs)
    plt.title(title, fontsize=14)
    return ax


# plot sunburst chart


def sunburst_chart(df, path=None, values=None, **plt_kwargs):
    """strip white spaces from records
    args:
        df: dataframe
        path: sunburst path
        values: chart values
        title: chart title
        **plt_kwargs: keyword args

    Returns:
        interative sunburst chart
    """
    fig = px.sunburst(df, path=path, values=values, **plt_kwargs)
    fig.update_traces(textinfo="label+percent entry")
    return fig


# plot bar chart


def bar_chart(df, col=None, **plt_kwargs):
    """plot bar chart
    args:
        df: dataframe
        col: columns to order chart
        values: chart values
        title: chart title
        **plt_kwargs: keyword args

    Returns:
        bar chart
    """

    try:
        df = df.sort_values(col, ascending=False)
    except:
        pass

    ax = df.plot(kind="bar", figsize=[10, 10], **plt_kwargs)

    return ax


# split solution readiness level 9 into "deployed in 1 market" and "deployed in multiple markets"


def split_SR9(project, deployment_level):
    """for projects with SR9, assign a new development level that splits SR9 into two
    args:
        project: case study project
        deployment_level: set development level that splits SR9

    Returns:
        new development level
    """

    project_index = case_studies[case_studies["Project"] == project].index[0]
    case_studies.iloc[
        project_index, case_studies.columns.get_loc("readiness_categories")
    ] = deployment_level


# %% [markdown]
# ### Read data

# %%
DATA_PATH = "../../../../crisis_intelligence"

# %%
case_studies = pd.read_csv(f"{DATA_PATH}/inputs/data/wp1_data_for_visualisation.csv")
case_study_table = pd.read_csv(f"{DATA_PATH}/inputs/data/case_study_table.csv")
CCI_case_studies = pd.read_csv(
    f"{DATA_PATH}/inputs/data/CCI_case_studies_innovation_stage.csv"
)

# %%
# strip white spaces from dataset and remove new lines

case_studies = strip_white_space(case_studies)
case_studies.replace(r"\s+|\\n", " ", regex=True, inplace=True)

case_study_table = strip_white_space(case_study_table)
case_study_table.replace(r"\s+|\\n", " ", regex=True, inplace=True)

CCI_case_studies = strip_white_space(CCI_case_studies)
CCI_case_studies.replace(r"\s+|\\n", " ", regex=True, inplace=True)

# %%
# count number of projects by humanitarian challenge

case_studies_freq = (
    case_studies.groupby("type_of_humanitarian_challenge")[["Project"]]
    .count()
    .reset_index()
)
case_studies_freq.rename(columns={"Project": "challenge_type_count"}, inplace=True)

# %%
# count number of projects by humanitarian challenge

case_studies_freq_high_level = (
    case_studies.groupby("high_level_crisis_types")[["Project"]].count().reset_index()
)
case_studies_freq_high_level.rename(
    columns={"Project": "challenge_type_count"}, inplace=True
)

# %% [markdown]
# ### Charts showing breakdown of case studies by type of crisis

# %%
# plot pie chart showing case study split by humanitarian challenge

ax = plt.subplots(figsize=[15, 10])
labels = case_studies_freq["type_of_humanitarian_challenge"]
title = "Spilt of case studies by humanitarian challenge"
pie(
    case_studies_freq["challenge_type_count"],
    ax=ax,
    labels=labels,
    title=title,
    colors=pal,
)

plt.savefig(f"{DATA_PATH}/outputs/figures/pie_chart_hum_crisis_split.png")

plt.show()


# %%
# plot pie chart showing the split of case studies by high level crises

ax = plt.subplots(figsize=[15, 10])
labels = case_studies_freq_high_level["high_level_crisis_types"]
title = "Spilt of case studies by high level humanitarian crisis"
pie(
    case_studies_freq_high_level["challenge_type_count"],
    ax=ax,
    labels=labels,
    title=title,
    colors=pal,
)

plt.savefig(f"{DATA_PATH}/outputs/figures/pie_chart_hum_crisis_split_v2.png")

plt.show()

# %% [markdown]
# ### Create sunburst chart showing crisis split

# %%
# group by high level and more granular categories of humanitarian crises

crisis_groupby = (
    case_studies.groupby(["high_level_crisis_types", "type_of_humanitarian_challenge"])[
        ["Project"]
    ]
    .count()
    .reset_index()
)

# %%
# replace more granuar categories for conflict, natural disaster, health and multiple with nan

crisis_groupby.loc[
    crisis_groupby["high_level_crisis_types"] == "Conflict",
    "type_of_humanitarian_challenge",
] = ""

crisis_groupby.loc[
    crisis_groupby["high_level_crisis_types"] == "Natural Disaster",
    "type_of_humanitarian_challenge",
] = ""

crisis_groupby.loc[
    crisis_groupby["high_level_crisis_types"] == "Health",
    "type_of_humanitarian_challenge",
] = ""

crisis_groupby.loc[
    crisis_groupby["high_level_crisis_types"] == "Multiple",
    "type_of_humanitarian_challenge",
] = ""

crisis_groupby = (
    crisis_groupby.groupby(
        ["high_level_crisis_types", "type_of_humanitarian_challenge"]
    )[["Project"]]
    .sum()
    .reset_index()
)

crisis_groupby = crisis_groupby.replace("", np.nan)


# %%
# plot sunburst chart

hum_crisis_split = sunburst_chart(
    crisis_groupby,
    path=["high_level_crisis_types", "type_of_humanitarian_challenge"],
    values="Project",
    color="Project",
    color_continuous_scale=pal,
    width=800,
    height=800,
)

hum_crisis_split.show()


# %% [markdown]
# ### Classification vs regression vs other broken down by supervised, unsupervised, weakly supervised and semi-supervised

# %%
# split classification_vs_regression column so there is a new record for each value in that column

case_studies_method_split = (
    case_studies.assign(
        classification_vs_regression=case_studies[
            "classification_vs_regression"
        ].str.split(",")
    )
    .explode("classification_vs_regression")
    .reset_index(drop=True)
)


# %%
# the 'Text Analytics for Resilience-Enabled Extreme Events Reconnaissance' project also contains multiple values
# in the method columnn
# further split values in method column so each method appears as a new record

recon_project_index = case_studies_method_split[
    case_studies_method_split["Project"]
    == "Text Analytics for Resilience-Enabled Extreme Events Reconnaissance"
].index[0]

case_studies_method_split.iloc[
    recon_project_index, case_studies_method_split.columns.get_loc("method")
] = "supervised"

case_studies_method_split.iloc[
    recon_project_index + 1, case_studies_method_split.columns.get_loc("method")
] = "unsupervised"

# %%
method_vs_class_vs_reg = (
    case_studies_method_split.groupby(["classification_vs_regression", "method"])[
        ["Project"]
    ]
    .count()
    .reset_index()
)

# %%
method_vs_class_vs_reg_pivot = method_vs_class_vs_reg.pivot(
    columns="method", index="classification_vs_regression", values="Project"
).fillna(0)

# %% [markdown]
# ### Create bar chart showing the split of methods by prediction types

# %%
# calculate the percentage split of methods by prediction types

method_vs_class_vs_reg_pivot = (
    method_vs_class_vs_reg_pivot / len(case_studies_method_split)
) * 100

# %%
# plot bar chart showing the split of methods by prediction types

ax = bar_chart(method_vs_class_vs_reg_pivot, "supervised", stacked=True, color=pal)
ax.set(xlabel="Prediction Type", ylabel="Percentage")
plt.savefig(f"{DATA_PATH}/outputs/figures/bar_chart_methods_split.png")
plt.show()

# %% [markdown]
# ### Create heatmap showing methods vs techniques

# %%
# split technique column so there is a new record for each value in that column

case_studies_techniques_split = (
    case_studies_method_split.assign(
        techniques=case_studies_method_split["techniques"].str.split(",")
    )
    .explode("techniques")
    .reset_index(drop=True)
)

# %%
case_studies_methods_vs_techniques = case_studies_techniques_split[
    ["method", "techniques"]
]

# %%
case_studies_methods_vs_techniques = strip_white_space(
    case_studies_methods_vs_techniques
)

# %%
methods_vs_techniques_heatmap = case_studies_methods_vs_techniques.pivot_table(
    index="techniques", columns="method", aggfunc=len, fill_value=0
)

# %%
# plot method vs techniques heatmeap

ax = plt.subplots(figsize=[10, 10]), sns.heatmap(
    methods_vs_techniques_heatmap, annot=True, cmap=pal
)
plt.savefig(f"{DATA_PATH}/outputs/figures/method_vs_techniques_heatmap.png")
plt.show()

# %% [markdown]
# ### Create solution readiness distribution chart

# %%
case_studies["solution_readiness_num_score"] = case_studies[
    "solution_readiness"
].str.slice(3, 4)
case_studies["solution_readiness_num_score"] = pd.to_numeric(
    case_studies["solution_readiness_num_score"]
)

# %%
# consolidate solution readiness levels

# map readiness levels 3 and 4 to proof of concept

case_studies.loc[
    (case_studies["solution_readiness_num_score"] == 3)
    | (case_studies["solution_readiness_num_score"] == 4),
    "readiness_categories",
] = "SRL 3/4: Concept/Idea"

# map readiness levels 5 and 6 to Prototype

case_studies.loc[
    (case_studies["solution_readiness_num_score"] == 5)
    | (case_studies["solution_readiness_num_score"] == 6),
    "readiness_categories",
] = "SRL 5/6: Prototype"

# map readiness level 7 pilot

case_studies.loc[
    (case_studies["solution_readiness_num_score"] == 7), "readiness_categories"
] = "SRL 7: Pilot"

# split solution readiness level 9 into "deployed in 1 market" and "deployed in multiple markets"

split_SR9("Sentry Syria", "SRL 9: Operational in one market")
split_SR9("Dataminr for humanitarian response", "SRL 9: Scaling to other markets")
split_SR9("RapiD (MapWithAI)", "SRL 9: Scaling to other markets")
split_SR9(
    "Common Social Accountability Platform (CSAP)", "SRL 9: Operational in one market"
)
split_SR9("AIME", "SRL 9: Scaling to other markets")
split_SR9("Companion Modelling", "SRL 9: Scaling to other markets")
split_SR9("Early Warning Project", "SRL 9: Scaling to other markets")
split_SR9("Internal-displacement", "SRL 9: Scaling to other markets")
split_SR9("Hunger map live", "SRL 9: Scaling to other markets")
split_SR9("InaSAFE - Flood impact model", "SRL 9: Operational in one market")
split_SR9("ViEWS: Violence Early-Warning System", "SRL 9: Scaling to other markets")

# %%
# append CCI case studies not part of the 33 PA case study analysis

case_studies_CCI_vs_PA = pd.merge(case_studies, CCI_case_studies, how="outer")

# %%
# merge case study table

case_studies_CCI_vs_PA = pd.merge(
    case_studies_CCI_vs_PA,
    case_study_table,
    how="left",
    left_on="Project",
    right_on="Title",
)

# %%
# create field for PA_vs_CCI

case_studies_CCI_vs_PA["PA_vs_CCI"] = np.where(
    case_studies_CCI_vs_PA["CCI Case Study"] == "N", "Predictive Analytics only", "CCI"
)

# %%
CCI_vs_PA = (
    case_studies_CCI_vs_PA.groupby(["PA_vs_CCI", "readiness_categories"])[["Project"]]
    .count()
    .reset_index()
)

# %%
CCI_vs_PA_pivot = CCI_vs_PA.pivot(
    columns="PA_vs_CCI", index="readiness_categories", values="Project"
).fillna(0)

# %%
# sort values based on maturity level

CCI_vs_PA_pivot.loc[
    CCI_vs_PA_pivot.index == "SRL 9: Scaling to other markets",
    "readiness_categories_num",
] = 5

CCI_vs_PA_pivot.loc[
    CCI_vs_PA_pivot.index == "SRL 9: Operational in one market",
    "readiness_categories_num",
] = 4

CCI_vs_PA_pivot.loc[
    CCI_vs_PA_pivot.index == "SRL 7: Pilot", "readiness_categories_num"
] = 3

CCI_vs_PA_pivot.loc[
    CCI_vs_PA_pivot.index == "SRL 5/6: Prototype", "readiness_categories_num"
] = 2

CCI_vs_PA_pivot.loc[
    CCI_vs_PA_pivot.index == "SRL 3/4: Concept/Idea", "readiness_categories_num"
] = 1

CCI_vs_PA_pivot.sort_values("readiness_categories_num", inplace=True)
CCI_vs_PA_pivot.drop("readiness_categories_num", axis=1, inplace=True)

# %%
# plot bar chart showing the split of CCI vs PA by readiness categories

color_map = ["#7F00FF", "#00aaba"]

ax = bar_chart(CCI_vs_PA_pivot, stacked=True, color=color_map)
ax.set(xlabel="Stage of innovation", ylabel="Number of case studies")
plt.savefig(f"{DATA_PATH}/outputs/figures/bar_chart_split_readiness_categories.png")
plt.show()

# %% [markdown]
# ### Create chart plotting the correlation between solution readiness and integration in humanitarian systems

# %%
# calculate number of case studies integrated into humanitarian workflows

hum_workflows = case_studies_CCI_vs_PA[
    ["Project", "PA_vs_CCI", "integrated_into_humanitarian_workflows_and_systems"]
]

# %%
# solution readiness vs integration in humanitarian systems heatmap

readiness_vs_integration = case_studies[
    ["solution_readiness", "integrated_into_humanitarian_workflows_and_systems"]
]


readiness_vs_integration_heatmap = readiness_vs_integration.pivot_table(
    index="solution_readiness",
    columns="integrated_into_humanitarian_workflows_and_systems",
    aggfunc=len,
    fill_value=0,
)


# %%
ax = plt.subplots(figsize=[10, 10]), sns.heatmap(
    readiness_vs_integration_heatmap, annot=True, cmap=pal2
).set(
    xlabel="Integrated into humanitarian workflows and systems",
    ylabel="Solution Readiness",
)
plt.savefig(f"{DATA_PATH}/outputs/figures/SR_vs_integration_heatmap.png")
plt.show()

# %% [markdown]
# ### Charts showing distribution of of algorithms

# %%
case_studies_algorithms_split = (
    case_studies_techniques_split.assign(
        algorithms=case_studies_techniques_split["algorithms"].str.split(",")
    )
    .explode("algorithms")
    .reset_index(drop=True)
)

# %%
case_studies_algorithms_split["AI"] = "Predictive Analytics"

# %%
case_studies_algorithms_split = strip_white_space(case_studies_algorithms_split)
algorithms_groupby = (
    case_studies_algorithms_split.groupby("algorithms")[["Project"]]
    .count()
    .reset_index()
    .sort_values("Project", ascending=False)
)
algorithms_groupby["Project"] = (
    algorithms_groupby["Project"] / len(algorithms_groupby)
) * 100


# %%
# plot bar chart of algorithm distribution

ax = algorithms_groupby.plot(
    kind="bar", x="algorithms", y="Project", figsize=(10, 8), color="purple"
)
ax.get_legend().remove()
ax.set(xlabel="Algorithm", ylabel="Percentage")
plt.savefig(f"{DATA_PATH}/outputs/figures/algo_dist.png")
plt.show()

# %%
algorithms_methods_groupby = (
    case_studies_algorithms_split.groupby(["AI", "method", "techniques", "algorithms"])[
        ["Project"]
    ]
    .count()
    .reset_index()
)

# %%
# plot sunburst chart of all algorithms

algo_sunburst = sunburst_chart(
    algorithms_methods_groupby,
    path=["AI", "method", "techniques", "algorithms"],
    color="method",
    width=1300,
    height=1300,
)

algo_sunburst.update_layout(uniformtext=dict(minsize=13, mode="show"))
algo_sunburst.update_traces(textinfo="label")


# %%

# %%

# %%

# %%

# %%
