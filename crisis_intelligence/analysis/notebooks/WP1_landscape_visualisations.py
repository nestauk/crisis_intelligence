# -*- coding: utf-8 -*-
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
# # CCI Landscape Visualisations for WP1

# %% [markdown]
# ### Import Libraries

# %%
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pylab import MaxNLocator


# %% [markdown]
# ### Functions

# %%
# plot pie chart


def pie(x, ax=None, labels=None, title=None, **plt_kwargs):
    """plot pie chart
    args:
        x: data
        ax: axis
        labels: pie chart labels
        title: chart title
        **plt_kwargs: keyword args

    Returns:
        pie chart
    """
    ax = ax
    plt.pie(x=x, autopct="%.1f%%", labels=labels, **plt_kwargs)
    plt.title(title, fontsize=14)
    return ax


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


# %% [markdown]
# ### Parameters for visualisations

# %%
# colour palette for charts

pal = sns.cubehelix_palette(as_cmap=False)
pal2 = sns.cubehelix_palette(as_cmap=True)

# %% [markdown]
# ### Read data

# %%
DATA_PATH = "../../../../crisis_intelligence"

# %%
case_studies = pd.read_csv(
    f"{DATA_PATH}/inputs/data/airtable_CCI_case_studies_7 July.csv"
)

# %% [markdown]
# ### Clean column names

# %%
case_studies.columns = case_studies.columns.str.replace("[()]", "")
case_studies.columns = case_studies.columns.str.replace(" ", "_")

# %% [markdown]
# ### Make directory

# %%
# make directory to store CCI landscape charts

if not os.path.exists(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape"):
    os.makedirs(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape")

# %% [markdown]
# ### Chart 1 - Regional distribution of CCI solutions
#

# %%
case_studies_region_breakdown = (
    case_studies.groupby("Region")["Name"].count().reset_index()
)

# %%
ax = plt.subplots(figsize=[15, 10])
labels = case_studies_region_breakdown["Region"]
title = "Regional distribution of CCI solutions"
pie(
    case_studies_region_breakdown["Name"],
    ax=ax,
    labels=labels,
    title=title,
    colors=pal,
)

plt.savefig(
    f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/pie_chart_regional_split.svg"
)

plt.show()

# %%
case_studies_region_breakdown.sort_values("Name", ascending=False, inplace=True)

ax = case_studies_region_breakdown.plot(
    kind="bar", x="Region", y="Name", figsize=(10, 8), color="purple"
)
ax.get_legend().remove()
ax.set(xlabel="Region", ylabel="Number of projects")

plt.savefig(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/bar_region_dist.svg")

plt.show()

# %% [markdown]
# ### Chart 2 - The types of organisations developing CCI solutions

# %%
case_studies_org_split = (
    case_studies.assign(
        Type_of_Organisation_developed=case_studies[
            "Type_of_Organisation_developed"
        ].str.split(",")
    )
    .explode("Type_of_Organisation_developed")
    .reset_index(drop=True)
)

# %%
case_studies_org_split_groupby = (
    case_studies_org_split.groupby("Type_of_Organisation_developed")["Name"]
    .count()
    .reset_index()
)
case_studies_org_split_groupby.sort_values("Name", inplace=True, ascending=False)

# %%
# plot bar chart of organisation (developed) distribution

ax = case_studies_org_split_groupby.plot(
    kind="bar",
    x="Type_of_Organisation_developed",
    y="Name",
    figsize=(10, 8),
    color="purple",
)
ax.get_legend().remove()
ax.set(xlabel="Type of organisation (developed)", ylabel="Number of projects")

plt.savefig(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/bar_chart_org_dist.svg")

plt.show()

# %% [markdown]
# ### Chart 3 - Chart showing the organisations/groups using CCI solutions

# %%

case_studies_using_org_split = (
    case_studies.assign(
        Type_of_Organisation_using=case_studies["Type_of_Organisation_using"].str.split(
            ","
        )
    )
    .explode("Type_of_Organisation_using")
    .reset_index(drop=True)
)

# %%
using_org_count = (
    case_studies_using_org_split.groupby("Type_of_Organisation_using")["Name"]
    .count()
    .reset_index()
)
using_org_count.rename(columns={"Name": "using_org_project_count"}, inplace=True)

# %%
using_org_vs_innovation_stage = (
    case_studies_using_org_split.groupby(
        ["Type_of_Organisation_using", "Stage_of_Innovation"]
    )[["Name"]]
    .count()
    .reset_index()
)

# %%
using_org_vs_innovation_stage_pivot = using_org_vs_innovation_stage.pivot(
    columns="Stage_of_Innovation", index="Type_of_Organisation_using", values="Name"
).fillna(0)

# %%
using_org_vs_innovation_stage_pivot = pd.merge(
    using_org_vs_innovation_stage_pivot,
    using_org_count,
    how="left",
    on="Type_of_Organisation_using",
)
using_org_vs_innovation_stage_pivot.sort_values(
    "using_org_project_count", ascending=False, inplace=True
)
using_org_vs_innovation_stage_pivot.drop(
    "using_org_project_count", axis=1, inplace=True
)
using_org_vs_innovation_stage_pivot.set_index(
    "Type_of_Organisation_using", inplace=True
)

# %%
# plot bar chart showing the split of using orgs by innovation stage

ax = bar_chart(
    using_org_vs_innovation_stage_pivot,
    "SRL 3/4: concept/idea",
    stacked=True,
    color=pal,
)
ax.set(xlabel="Type of Organisation (using)", ylabel="Number of projects")

plt.savefig(
    f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/bar_chart_innovation_split.svg"
)

plt.show()

# %% [markdown]
# ### Chart 4 - Chart showing the types of people contributing to CCI solutions

# %%

case_studies_people_split = (
    case_studies.assign(
        People_contributing=case_studies["People_contributing"].str.split(",")
    )
    .explode("People_contributing")
    .reset_index(drop=True)
)

# %%
case_studies_people_split_groupby = (
    case_studies_people_split.groupby("People_contributing")["Name"]
    .count()
    .reset_index()
)
case_studies_people_split_groupby.sort_values("Name", inplace=True, ascending=False)

# %%
# plot bar chart of People (contributing) distribution

ax = case_studies_people_split_groupby.plot(
    kind="bar", x="People_contributing", y="Name", figsize=(10, 8), color="purple"
)
ax.get_legend().remove()
ax.set(xlabel="People (contributing)", ylabel="Number of projects")

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/bar_chart_people_dist.svg")

plt.show()

# %% [markdown]
# ### Chart 5 - Chart showing the application of CCI solutions across the crisis management cycle

# %%
case_studies_response_split = (
    case_studies.assign(
        Stage_of_crisis_response=case_studies["Stage_of_crisis_response"].str.split(",")
    )
    .explode("Stage_of_crisis_response")
    .reset_index(drop=True)
)

# %%
response_vs_crisis_type = (
    case_studies_response_split.groupby(["Stage_of_crisis_response", "Type_of_Crisis"])[
        "Name"
    ]
    .count()
    .reset_index()
)

# %%
response_vs_crisis_type_pivot = response_vs_crisis_type.pivot(
    columns="Type_of_Crisis", index="Stage_of_crisis_response", values="Name"
).fillna(0)

# %%

response_vs_crisis_type_pivot.loc[
    response_vs_crisis_type_pivot.index == "mitigation", "order"
] = 1

response_vs_crisis_type_pivot.loc[
    response_vs_crisis_type_pivot.index == "preparedness", "order"
] = 2

response_vs_crisis_type_pivot.loc[
    response_vs_crisis_type_pivot.index == "response", "order"
] = 3

response_vs_crisis_type_pivot.loc[
    response_vs_crisis_type_pivot.index == "recovery", "order"
] = 4


# %%
response_vs_crisis_type_pivot.sort_values("order", ascending=True, inplace=True)
response_vs_crisis_type_pivot.drop("order", axis=1, inplace=True)

# %%
# plot bar chart showing the split reponse stage by type of crisis

ax = bar_chart(response_vs_crisis_type_pivot, stacked=True, color=pal)
ax.set(xlabel="Type of Organisation (using)", ylabel="Number of projects")

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(
    f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/bar_chart_crisis_type_split.svg"
)

plt.show()

# %% [markdown]
# ### Chart 6 - Chart showing data sources being used by CCI solutions

# %%
case_studies_data_level_2_split = (
    case_studies.assign(Data_level_2=case_studies["Data_level_2"].str.split(","))
    .explode("Data_level_2")
    .reset_index(drop=True)
)

# %%
data_level_2_count = (
    case_studies_data_level_2_split.groupby("Data_level_2")["Name"]
    .count()
    .reset_index()
)
data_level_2_count.sort_values("Name", inplace=True, ascending=False)

# %%
# plot data_level_2 distribution

ax = data_level_2_count.plot(
    kind="bar", x="Data_level_2", y="Name", figsize=(10, 8), color="purple"
)
ax.get_legend().remove()
ax.set(xlabel="Data Level 2", ylabel="Number of projects")

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/data_level_2_dist.svg")

plt.show()

# %%
# plot data_level_2 distribution - only consider cases with more than 2 projects

ax = data_level_2_count[data_level_2_count["Name"] >= 2].plot(
    kind="bar", x="Data_level_2", y="Name", figsize=(10, 8), color="purple"
)
ax.get_legend().remove()
ax.set(xlabel="Data Level 2", ylabel="Number of projects")

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/data_level_2_dist_v2.svg")

plt.show()

# %% [markdown]
# ### Chart 7 - Chart showing types of methods being used by CCI solutions

# %%
case_studies_method_split = (
    case_studies.assign(Method=case_studies["Method"].str.split(","))
    .explode("Method")
    .reset_index(drop=True)
)

# %%
method_count = case_studies_method_split.groupby("Method")["Name"].count().reset_index()
method_count.sort_values("Name", inplace=True, ascending=False)

# %%
# plot method distribution

ax = method_count.plot(
    kind="bar", x="Method", y="Name", figsize=(10, 8), color="purple"
)
ax.get_legend().remove()
ax.set(xlabel="Method", ylabel="Number of projects")

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/bar_chart_method_dist.svg")

plt.show()

# %%
# plot method distribution - only consider cases with more than 2 projects

ax = method_count[method_count["Name"] >= 2].plot(
    kind="bar", x="Method", y="Name", figsize=(10, 8), color="purple"
)
ax.get_legend().remove()
ax.set(xlabel="Method", ylabel="Number of projects")

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(
    f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/bar_chart_method_dist_v2.svg"
)

plt.show()

# %% [markdown]
# ### Chart 8 - Impact and accuracy/effectiveness of CCI solutions

# %%
impact_count = case_studies.groupby("Impact_Measured?")["Name"].count().reset_index()
impact_count.sort_values("Name", inplace=True, ascending=False)

# %%
# plot impact distribution

ax = impact_count.plot(
    kind="bar", x="Impact_Measured?", y="Name", figsize=(10, 8), color="purple"
)
ax.get_legend().remove()
ax.set(xlabel="Impact_Measured?", ylabel="Number of projects")

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(f"{DATA_PATH}/outputs/figures/WP1_CCI_landscape/bar_chart_impact_dist.svg")

plt.show()

# %%

# %%
