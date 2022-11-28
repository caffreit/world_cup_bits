# world_cup_bits
A world cup (2022) sim based on Elo scores. Ran in conjunction with a pool based around point earned. So I was interested in keeping track of coins too.

```python
import pandas as pd
import numpy as np
import random
import scipy.stats as ss

import datetime
import time
from datetime import datetime
from datetime import timedelta

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objects as go # Import the graphical object

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#from IPython.core.pylabtools import figsize

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 40em; }</style>"))

pd.set_option('display.max_columns', 273)
pd.set_option('display.max_rows', 273)
```


<style>.container { width:95% !important; }</style>



<style>div.output_scroll { height: 40em; }</style>



```python
countries = ['Brazil','Belgium','Argentina','France','England','Spain','Netherlands','Portugal',
'Denmark','Germany','Croatia','Mexico','Uruguay','Switzerland','United_States','Senegal',
'Wales','Iran','Serbia','Morocco','Japan','Poland','Korea','Tunisia','Costa_Rica','Australia',
'Canada','Cameroon','Ecuador','Qatar','Saudi_Arabia','Ghana']

countries_interest = ['Brazil','Argentina','England','Netherlands','Croatia','Uruguay','Wales','Ecuador','Qatar']

elos = [2169,2025,2143,2005,1920,2045,2040,2004,1971,1963,1927,1809,
1936,1902,1798,1687,1790,1797,1892,1753,1787,1814,1786,1707,1743,
1719,1776,1609,1833,1680,1635,1567]

coins_per_win = [1,1,1,1,2,2,2,2,4,4,4,4,8,8,8,8,16,16,16,16,32,32,32,32,64,64,64,64,128,128,128,128]

price = [75,26,60,56,45,50,36,30,23,39,17,12,17,12,11,12,11,8,13,9,9,10,9,8,8,8,9,9,9,8,8,9]

groups = [
['Qatar', 'Ecuador', 'Senegal', 'Netherlands'],
['England', 'Iran', 'United_States', 'Wales'],
['Argentina', 'Saudi_Arabia', 'Mexico', 'Poland'],
['France', 'Australia', 'Denmark', 'Tunisia'],
['Spain', 'Costa_Rica', 'Germany', 'Japan'],
['Belgium', 'Canada', 'Morocco', 'Croatia'],
['Brazil', 'Serbia', 'Switzerland', 'Cameroon'],
['Portugal', 'Ghana', 'Uruguay', 'Korea']]

letters = ['A','B','C','D','E','F','G','H']

group_winners = ['A1','A2','B1','B2','C1','C2','D1','D2','E1','E2','F1','F2','G1','G2','H1','H2']

tier_name_list = ['GROUP','R16','QF','SF_W','SF_L','FOURTH','THIRD','SECOND','WINNER']

# dictionary of Elo scores
# count of coins retunred by a team per win

dic_elos = dict(zip(countries, elos))
dic_coin_win = dict(zip(countries, coins_per_win))
dic_groups = dict(zip(letters, groups))
dic_price = dict(zip(countries, price))
```


```python
df = pd.read_csv('team_progress.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>result</th>
      <th>country</th>
      <th>tier</th>
      <th>sim_no</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>A1</td>
      <td>Netherlands</td>
      <td>GROUP</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A2</td>
      <td>Senegal</td>
      <td>GROUP</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>B1</td>
      <td>England</td>
      <td>GROUP</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>B2</td>
      <td>Wales</td>
      <td>GROUP</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>C1</td>
      <td>Argentina</td>
      <td>GROUP</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# counts data Sankey


```python
country = 'Ecuador'
group = 'A'
values = []

# prob exiting the groups
# prob of C1 or C2

df_group_1 = df[(df.country == country)&(df.result == group+'1')]
df_group_2 = df[(df.country == country)&(df.result == group+'2')]
values.append(df.sim_no.nunique() - (df_group_1.sim_no.nunique() + df_group_2.sim_no.nunique()))
values.append(df_group_1.sim_no.nunique())
values.append(df_group_2.sim_no.nunique())

# when country is C1, how likely are they to win
# of all the times a country is C1, how many times did they reach R16

df_R16_w_1 = df[(df.country == country)&(df.tier == 'R16')&(df.sim_no.isin(df_group_1.sim_no.unique()))]

values.append(df_R16_w_1.sim_no.nunique())
values.append(df_group_1.sim_no.nunique() - df_R16_w_1.sim_no.nunique())

# when country is C2, how likely are they to win
# of all the times a country is C2, how many times did they reach R16

df_R16_w_2 = df[(df.country == country)&(df.tier == 'R16')&(df.sim_no.isin(df_group_2.sim_no.unique()))]

values.append(df_R16_w_2.sim_no.nunique())
values.append(df_group_2.sim_no.nunique() - df_R16_w_2.sim_no.nunique())

# if they reach QF2, how likely are they to win

df_QF_w_1 = df[(df.tier == 'QF')&(df.country == country)&(df.sim_no.isin(df_R16_w_1.sim_no.unique()))]

values.append(df_QF_w_1.sim_no.nunique())
values.append(df_R16_w_1.sim_no.nunique() - df_QF_w_1.sim_no.nunique())

# if they reach QF4, how likely are they to win
# they are now in the semis

df_QF_w_2 = df[(df.tier == 'QF')&(df.country == country)&(df.sim_no.isin(df_R16_w_2.sim_no.unique()))]

values.append(df_QF_w_2.sim_no.nunique())
values.append(df_R16_w_2.sim_no.nunique() - df_QF_w_2.sim_no.nunique())

# if they reach SF1, how likely are they to win

df_SF_w_1 = df[(df.tier == 'SF_W')&(df.country == country)&(df.sim_no.isin(df_QF_w_1.sim_no.unique()))]
values.append(df_SF_w_1.sim_no.nunique())
values.append(df_QF_w_1.sim_no.nunique() - df_SF_w_1.sim_no.nunique())

# if they reach SF2, how likely are they to win

df_SF_w_2 = df[(df.tier == 'SF_W')&(df.country == country)&(df.sim_no.isin(df_QF_w_2.sim_no.unique()))]
values.append(df_SF_w_2.sim_no.nunique())
values.append(df_QF_w_2.sim_no.nunique() - df_SF_w_2.sim_no.nunique())

values.append(df[(df.country == country) & (df.tier == 'WINNER')].sim_no.nunique())
values.append(df[(df.country == country) & (df.tier == 'SECOND')].sim_no.nunique())
values.append(df[(df.country == country) & (df.tier == 'THIRD')].sim_no.nunique())
values.append(df[(df.country == country) & (df.tier == 'FOURTH')].sim_no.nunique())

node_label = ["GROUPS", "KO", "C1", "C2", "QF2", "QF4", "SF1", "SF2", "FINAL", "RUNNER_UP", "FIRST", "SECOND", "THIRD", "FOURTH"]
node_dict = {y:x for x, y in enumerate(node_label)}

source = ['GROUPS','GROUPS','GROUPS','C1', 'C1', 'C2', 'C2', 'QF2', 'QF2','QF4', 'QF4','SF1',  'SF1',      'SF2',  'SF2',       'FINAL', 'FINAL',  'RUNNER_UP','RUNNER_UP']
target = ['KO',    'C1',    'C2',    'QF2','KO', 'QF4','KO', 'SF1', 'KO', 'SF2', 'KO', 'FINAL','RUNNER_UP','FINAL','RUNNER_UP', 'FIRST', 'SECOND', 'THIRD',    'FOURTH'] 

source_node = [node_dict[x] for x in source]
target_node = [node_dict[x] for x in target]

fig = go.Figure( 
    data=[go.Sankey( # The plot we are interest
        # This part is for the node information
        node = dict( 
            label = node_label
        ),
        # This part is for the link information
        link = dict(
            source = source_node,
            target = target_node,
            value = values
        ))])

# With this save the plots 
plot(fig,
     image_filename='sankey_counts_'+country, 
     image='png', 
     image_width=1000, 
     image_height=600
)
fig.update_layout(title_text=country,
                  font_size=15)

# And shows the plot
fig.show()
```


<div>


            <div id="027d808b-ad3a-45cf-b775-6622e91d56ab" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("027d808b-ad3a-45cf-b775-6622e91d56ab")) {
                    Plotly.newPlot(
                        '027d808b-ad3a-45cf-b775-6622e91d56ab',
                        [{"link": {"source": [0, 0, 0, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9], "target": [1, 2, 3, 4, 1, 5, 1, 6, 1, 7, 1, 8, 9, 8, 9, 10, 11, 12, 13], "value": [5005, 2722, 7273, 1417, 1305, 3266, 4007, 306, 1111, 1153, 2113, 70, 236, 691, 462, 133, 628, 174, 524]}, "node": {"label": ["GROUPS", "KO", "C1", "C2", "QF2", "QF4", "SF1", "SF2", "FINAL", "RUNNER_UP", "FIRST", "SECOND", "THIRD", "FOURTH"]}, "type": "sankey"}],
                        {"font": {"size": 15}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Ecuador"}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('027d808b-ad3a-45cf-b775-6622e91d56ab');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
